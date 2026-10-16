use crate::SimpleDate;

pub trait MarketCalendar: Send + Sync {
    fn days_in_year(&self) -> f64;
    fn count_days(&self, start: SimpleDate, end: SimpleDate) -> i64;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ActualCalendar;

impl MarketCalendar for ActualCalendar {
    fn days_in_year(&self) -> f64 {
        365.0
    }

    fn count_days(&self, start: SimpleDate, end: SimpleDate) -> i64 {
        start.days_between(&end).max(0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeekdayCalendar {
    days_in_year: f64,
}

impl WeekdayCalendar {
    pub fn new(days_in_year: f64) -> Self {
        Self { days_in_year }
    }
}

impl Default for WeekdayCalendar {
    fn default() -> Self {
        Self {
            days_in_year: 252.0,
        }
    }
}

impl MarketCalendar for WeekdayCalendar {
    fn days_in_year(&self) -> f64 {
        self.days_in_year
    }

    fn count_days(&self, start: SimpleDate, end: SimpleDate) -> i64 {
        let total = start.days_between(&end);
        if total <= 0 {
            return 0;
        }

        let mut business_days = 0_i64;
        for offset in 0..total {
            let d = start.add_days(offset);
            let wd = d.weekday_mon0();
            if wd < 5 {
                business_days += 1;
            }
        }
        business_days
    }
}

pub trait HolidayCalendar: Send + Sync {
    fn is_business_day(&self, d: SimpleDate) -> bool;

    /// Business days in `[start, end]`.
    fn business_days(&self, start: SimpleDate, end: SimpleDate) -> Vec<SimpleDate> {
        (0..=start.days_between(&end))
            .map(|i| start.add_days(i))
            .filter(|&d| self.is_business_day(d))
            .collect()
    }

    /// The `n`-th business day strictly after `d`; `d` itself when `n == 0`.
    fn advance(&self, mut d: SimpleDate, n: u32) -> SimpleDate {
        for _ in 0..n {
            d = d.add_days(1);
            while !self.is_business_day(d) {
                d = d.add_days(1);
            }
        }
        d
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WeekendsOnly;

impl HolidayCalendar for WeekendsOnly {
    fn is_business_day(&self, d: SimpleDate) -> bool {
        d.weekday_mon0() < 5
    }
}

#[derive(Debug, Clone, Default)]
pub struct HolidayList(Vec<SimpleDate>);

impl HolidayList {
    pub fn new(mut holidays: Vec<SimpleDate>) -> Self {
        holidays.sort_unstable();
        holidays.dedup();
        Self(holidays)
    }
}

impl HolidayCalendar for HolidayList {
    fn is_business_day(&self, d: SimpleDate) -> bool {
        WeekendsOnly.is_business_day(d) && self.0.binary_search(&d).is_err()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusinessDayConvention {
    Following,
    ModifiedFollowing,
    Preceding,
}

impl BusinessDayConvention {
    pub fn adjust(self, cal: &impl HolidayCalendar, d: SimpleDate) -> SimpleDate {
        let roll = |step| {
            let mut x = d;
            while !cal.is_business_day(x) {
                x = x.add_days(step);
            }
            x
        };
        match self {
            Self::Following => roll(1),
            Self::Preceding => roll(-1),
            Self::ModifiedFollowing => Some(roll(1))
                .filter(|f| f.month == d.month)
                .unwrap_or_else(|| roll(-1)),
        }
    }
}
