use std::{
    collections::{BTreeMap, HashMap, VecDeque, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

use rayon::prelude::*;

use crate::{FinancialProduct, PricingEngine, QuantLibError, StochasticModel};

pub fn make_cache_key<P: FinancialProduct, M: StochasticModel>(
    product: &P,
    model: &M,
    kwargs: &BTreeMap<String, String>,
) -> Result<String, QuantLibError> {
    let product_repr = flatten_map(&product.get_parameters());
    let model_repr = flatten_map(&model.get_parameters());
    let kwargs_repr = flatten_map(kwargs);
    let payload = format!("product:{product_repr}|model:{model_repr}|kwargs:{kwargs_repr}");

    let mut hasher = DefaultHasher::new();
    payload.hash(&mut hasher);
    let hash = hasher.finish();
    Ok(format!("{hash:016x}"))
}

#[derive(Debug)]
struct SimpleLruCache {
    maxsize: usize,
    values: HashMap<String, f64>,
    order: VecDeque<String>,
}

impl SimpleLruCache {
    fn new(maxsize: usize) -> Self {
        Self {
            maxsize: maxsize.max(1),
            values: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<f64> {
        let value = self.values.get(key).copied()?;
        self.touch_key(key);
        Some(value)
    }

    fn put(&mut self, key: String, value: f64) {
        if self.values.contains_key(&key) {
            self.values.insert(key.clone(), value);
            self.touch_key(&key);
            return;
        }

        if self.values.len() >= self.maxsize
            && let Some(evicted) = self.order.pop_front()
        {
            self.values.remove(&evicted);
        }

        self.order.push_back(key.clone());
        self.values.insert(key, value);
    }

    fn touch_key(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key)
            && let Some(found) = self.order.remove(pos)
        {
            self.order.push_back(found);
        }
    }
}

#[derive(Debug)]
pub struct OptionValuationContext<E> {
    engine: E,
    cache_enabled: bool,
    parallel: bool,
    cache: Option<Arc<Mutex<SimpleLruCache>>>,
}

impl<E> OptionValuationContext<E> {
    pub fn new(engine: E, cache_enabled: bool, cache_maxsize: usize, parallel: bool) -> Self {
        let cache = if cache_enabled {
            Some(Arc::new(Mutex::new(SimpleLruCache::new(cache_maxsize))))
        } else {
            None
        };

        Self {
            engine,
            cache_enabled,
            parallel,
            cache,
        }
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }

    pub fn value_option<P, M>(
        &self,
        product: &P,
        model: &M,
        use_cache: Option<bool>,
        kwargs: &BTreeMap<String, String>,
    ) -> Result<f64, QuantLibError>
    where
        E: PricingEngine<P, M>,
        P: FinancialProduct,
        M: StochasticModel,
    {
        let cache_flag = use_cache.unwrap_or(self.cache_enabled);
        let key = if cache_flag {
            Some(make_cache_key(product, model, kwargs)?)
        } else {
            None
        };

        if let (Some(cache), Some(key)) = (&self.cache, key.as_ref())
            && let Ok(mut guard) = cache.lock()
            && let Some(value) = guard.get(key)
        {
            return Ok(value);
        }

        let price = self.engine.calculate_price(product, model)?;

        if let (Some(cache), Some(key)) = (&self.cache, key)
            && let Ok(mut guard) = cache.lock()
        {
            guard.put(key, price);
        }

        Ok(price)
    }

    pub fn value_options<P, M>(
        &self,
        products: &[P],
        model: &M,
        use_cache: Option<bool>,
        kwargs: &BTreeMap<String, String>,
    ) -> Result<Vec<f64>, QuantLibError>
    where
        E: PricingEngine<P, M> + Sync,
        P: FinancialProduct + Sync,
        M: StochasticModel + Sync,
    {
        if self.parallel {
            let prices: Vec<Result<f64, QuantLibError>> = products
                .par_iter()
                .map(|product| self.value_option(product, model, use_cache, kwargs))
                .collect();
            prices.into_iter().collect()
        } else {
            products
                .iter()
                .map(|product| self.value_option(product, model, use_cache, kwargs))
                .collect()
        }
    }
}

fn flatten_map(map: &BTreeMap<String, String>) -> String {
    map.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(";")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        FinancialProduct, OptionValuationContext, PricingEngine, QuantLibError, StochasticModel,
        make_cache_key,
    };

    struct DummyProduct;

    impl FinancialProduct for DummyProduct {
        fn payoff(&self, _spot_prices: &[f64]) -> Vec<f64> {
            vec![]
        }

        fn get_parameters(&self) -> BTreeMap<String, String> {
            BTreeMap::from([("k".to_owned(), "100.0".to_owned())])
        }
    }

    struct DummyModel;

    impl StochasticModel for DummyModel {
        fn get_parameters(&self) -> BTreeMap<String, String> {
            BTreeMap::from([("sigma".to_owned(), "0.2".to_owned())])
        }
    }

    #[derive(Clone, Copy)]
    struct DummyEngine;

    impl PricingEngine<DummyProduct, DummyModel> for DummyEngine {
        fn calculate_price(
            &self,
            _product: &DummyProduct,
            _model: &DummyModel,
        ) -> Result<f64, QuantLibError> {
            Ok(42.0)
        }
    }

    #[test]
    fn cache_key_is_stable() {
        let kwargs = BTreeMap::from([("foo".to_owned(), "bar".to_owned())]);
        let p = DummyProduct;
        let m = DummyModel;

        let key_1 = make_cache_key(&p, &m, &kwargs).expect("cache key");
        let key_2 = make_cache_key(&p, &m, &kwargs).expect("cache key");

        assert_eq!(key_1, key_2);
    }

    #[test]
    fn value_options_parallel_matches_serial() {
        let products = vec![DummyProduct, DummyProduct, DummyProduct];
        let model = DummyModel;
        let kwargs = BTreeMap::new();

        let serial = OptionValuationContext::new(DummyEngine, true, 16, false);
        let parallel = OptionValuationContext::new(DummyEngine, true, 16, true);

        let serial_prices = serial
            .value_options(&products, &model, None, &kwargs)
            .expect("serial prices");
        let parallel_prices = parallel
            .value_options(&products, &model, None, &kwargs)
            .expect("parallel prices");

        assert_eq!(serial_prices, parallel_prices);
        assert_eq!(parallel_prices, vec![42.0, 42.0, 42.0]);
    }
}
