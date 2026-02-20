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
