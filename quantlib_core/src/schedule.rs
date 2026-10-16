use crate::calendar::{BusinessDayConvention, HolidayCalendar, MarketCalendar};
use crate::simple_date::days_in_month;
use crate::{QuantLibError, SimpleDate};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    fixings: Vec<SimpleDate>,
    payments: Vec<SimpleDate>,
    /// pay_index[k] = payment slot that fixing k settles in
    pay_index: Vec<usize>,
}

impl Schedule {
    pub fn fixings(&self) -> &[SimpleDate] {
        &self.fixings
    }

    pub fn payments(&self) -> &[SimpleDate] {
        &self.payments
    }

    pub fn pay_index(&self) -> &[usize] {
        &self.pay_index
    }

    pub fn new(
        fixings: Vec<SimpleDate>,
        payments: Vec<SimpleDate>,
        pay_index: Vec<usize>,
    ) -> Result<Self, QuantLibError> {
        let err = |msg: &str| Err(QuantLibError::Validation(msg.to_owned()));
        if fixings.is_empty() || pay_index.len() != fixings.len() {
            return err("need one pay_index per fixing and at least one fixing");
        }
        let increasing = |ds: &[SimpleDate]| ds.windows(2).all(|w| w[0] < w[1]);
        if !increasing(&fixings) || !increasing(&payments) {
            return err("fixings and payments must be strictly increasing");
        }
        if fixings
            .iter()
            .zip(&pay_index)
            .any(|(f, &p)| payments.get(p).is_none_or(|pay| pay < f))
        {
            return err("every fixing needs a payment on or after it");
        }
        Ok(Self {
            fixings,
            payments,
            pay_index,
        })
    }

    /// Groups fixings by calendar month; each pays `lag` business days after the adjusted month end.
    pub fn monthly(
        fixings: Vec<SimpleDate>,
        cal: &impl HolidayCalendar,
        convention: BusinessDayConvention,
        lag: u32,
    ) -> Result<Self, QuantLibError> {
        let mut payments = Vec::new();
        let mut last = None;
        let pay_index = fixings
            .iter()
            .map(|f| {
                if last != Some((f.year, f.month)) {
                    last = Some((f.year, f.month));
                    let end = SimpleDate {
                        day: days_in_month(f.year, f.month),
                        ..*f
                    };
                    payments.push(cal.advance(convention.adjust(cal, end), lag));
                }
                payments.len() - 1
            })
            .collect();
        Self::new(fixings, payments, pay_index)
    }

    /// (fixing times, payment times) as `count_days(valuation, d) / days_in_year`; dates on or before valuation map to 0.
    pub fn times(&self, valuation: SimpleDate, cal: &dyn MarketCalendar) -> (Vec<f64>, Vec<f64>) {
        let t = |ds: &[SimpleDate]| {
            ds.iter()
                .map(|&d| cal.count_days(valuation, d) as f64 / cal.days_in_year())
                .collect()
        };
        (t(&self.fixings), t(&self.payments))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendar::{BusinessDayConvention::*, HolidayList};

    fn d(s: &str) -> SimpleDate {
        SimpleDate::parse(s).unwrap()
    }

    // Fri, Mon, Wed, Mon, Sat, Fri; unsorted and with a duplicate on purpose.
    const HOLIDAYS: [&str; 7] = [
        "2027-03-31",
        "2027-12-31",
        "2027-01-01",
        "2027-03-29",
        "2027-03-31",
        "2027-05-03",
        "2027-12-25",
    ];

    fn holidays() -> HolidayList {
        HolidayList::new(HOLIDAYS.map(d).to_vec())
    }

    #[test]
    fn daily_2027_skips_weekends_and_holidays() {
        let hols = HOLIDAYS.map(d);
        let start = d("2027-01-01");
        let brute: Vec<_> = (0..365)
            .map(|i| start.add_days(i))
            .filter(|x| x.weekday_mon0() < 5 && !hols.contains(x))
            .collect();
        assert_eq!(brute.len(), 261 - 5);
        assert_eq!(holidays().business_days(start, d("2027-12-31")), brute);
    }

    #[test]
    fn conventions_roll_weekends_and_month_ends() {
        let cal = holidays();
        for (date, conv, want) in [
            ("2027-01-16", Following, "2027-01-18"),
            ("2027-01-16", ModifiedFollowing, "2027-01-18"),
            ("2027-01-16", Preceding, "2027-01-15"),
            ("2027-01-31", Following, "2027-02-01"),
            ("2027-01-31", ModifiedFollowing, "2027-01-29"),
            ("2027-01-31", Preceding, "2027-01-29"),
        ] {
            assert_eq!(conv.adjust(&cal, d(date)), d(want), "{date} {conv:?}");
        }
    }

    #[test]
    fn monthly_assigns_fixings_to_months_and_pays_after_lag() {
        let cal = holidays();
        let fixings = cal.business_days(d("2027-01-01"), d("2027-12-31"));
        let s = Schedule::monthly(fixings, &cal, ModifiedFollowing, 2).unwrap();
        assert_eq!(s.payments().len(), 12);
        assert!(
            s.fixings()
                .iter()
                .zip(s.pay_index())
                .all(|(f, &p)| p == f.month as usize - 1)
        );
        for (month, want) in [
            (1, "2027-02-02"),  // Sun 31st -> Fri 29th, +2
            (3, "2027-04-02"),  // holiday 31st -> Tue 30th, +2 skipping the 31st
            (4, "2027-05-05"),  // Fri 30th, +2 skipping holiday May 3rd
            (12, "2028-01-04"), // holiday 31st -> Thu 30th, +2 across year end
        ] {
            assert_eq!(s.payments()[month - 1], d(want), "month {month}");
        }
        let (tf, tp) = s.times(d("2027-06-01"), &crate::ActualCalendar);
        assert_eq!((tf[0], tp[11]), (0.0, 217.0 / 365.0));
    }

    #[test]
    fn rejects_bad_schedules() {
        let (a, b) = (d("2027-01-04"), d("2027-01-05"));
        for (fixings, payments, pay_index) in [
            (vec![b, a], vec![b], vec![0, 0]),
            (vec![a, a], vec![b], vec![0, 0]),
            (vec![a, b], vec![a], vec![0, 0]),
            (vec![a, b], vec![b, a], vec![0, 0]),
            (vec![a, b], vec![b], vec![0]),
            (vec![a, b], vec![b], vec![0, 1]),
        ] {
            assert!(Schedule::new(fixings, payments, pay_index).is_err());
        }
    }
}
