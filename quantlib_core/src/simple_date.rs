use core::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::QuantLibError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SimpleDate {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

impl SimpleDate {
    pub fn new(year: i32, month: u32, day: u32) -> Result<Self, QuantLibError> {
        if !(1..=12).contains(&month) {
            return Err(QuantLibError::DateParse(format!("invalid month: {month}")));
        }

        let max_day = days_in_month(year, month);
        if day == 0 || day > max_day {
            return Err(QuantLibError::DateParse(format!(
                "invalid day: {day} for {year:04}-{month:02}"
            )));
        }

        Ok(Self { year, month, day })
    }

    pub fn parse(value: &str) -> Result<Self, QuantLibError> {
        parse_ymd(value).or_else(|_| parse_dmy(value))
    }

    pub fn format_ymd(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    pub fn days_since_epoch(&self) -> i64 {
        days_from_civil(self.year, self.month, self.day)
    }

    pub fn days_between(&self, other: &Self) -> i64 {
        other.days_since_epoch() - self.days_since_epoch()
    }

    pub fn add_days(&self, days: i64) -> Self {
        civil_from_days(self.days_since_epoch() + days)
    }

    pub fn weekday_mon0(&self) -> i64 {
        (self.days_since_epoch() + 3).rem_euclid(7)
    }

    pub fn today_utc() -> Self {
        let now = SystemTime::now();
        let days = match now.duration_since(UNIX_EPOCH) {
            Ok(duration) => (duration.as_secs() / 86_400) as i64,
            Err(error) => -((error.duration().as_secs() / 86_400) as i64),
        };
        civil_from_days(days)
    }
}

impl fmt::Display for SimpleDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_ymd())
    }
}

fn parse_ymd(value: &str) -> Result<SimpleDate, QuantLibError> {
    let parts: Vec<&str> = value.split('-').collect();
    if parts.len() != 3 || parts[0].len() != 4 {
        return Err(QuantLibError::DateParse("expected YYYY-MM-DD".to_owned()));
    }

    let year = parts[0]
        .parse::<i32>()
        .map_err(|_| QuantLibError::DateParse("invalid year".to_owned()))?;
    let month = parts[1]
        .parse::<u32>()
        .map_err(|_| QuantLibError::DateParse("invalid month".to_owned()))?;
    let day = parts[2]
        .parse::<u32>()
        .map_err(|_| QuantLibError::DateParse("invalid day".to_owned()))?;

    SimpleDate::new(year, month, day)
}

fn parse_dmy(value: &str) -> Result<SimpleDate, QuantLibError> {
    let parts: Vec<&str> = value.split('-').collect();
    if parts.len() != 3 || parts[2].len() != 4 {
        return Err(QuantLibError::DateParse("expected DD-MM-YYYY".to_owned()));
    }

    let day = parts[0]
        .parse::<u32>()
        .map_err(|_| QuantLibError::DateParse("invalid day".to_owned()))?;
    let month = parts[1]
        .parse::<u32>()
        .map_err(|_| QuantLibError::DateParse("invalid month".to_owned()))?;
    let year = parts[2]
        .parse::<i32>()
        .map_err(|_| QuantLibError::DateParse("invalid year".to_owned()))?;

    SimpleDate::new(year, month, day)
}

fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

pub(crate) fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let y = year - i32::from(month <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = month as i32;
    let d = day as i32;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146_097 + doe as i64 - 719_468
}

fn civil_from_days(days_since_epoch: i64) -> SimpleDate {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe as i32 + era as i32 * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mut m = mp + if mp < 10 { 3 } else { -9 };
    y += i32::from(m <= 2);
    if m <= 0 {
        m += 12;
    }

    SimpleDate {
        year: y,
        month: m as u32,
        day: d as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::SimpleDate;

    #[test]
    fn parses_and_formats_dates() {
        let date = SimpleDate::parse("2026-02-10").expect("valid date");
        assert_eq!(date.format_ymd(), "2026-02-10");

        let date_dmy = SimpleDate::parse("10-02-2026").expect("valid dmy date");
        assert_eq!(date_dmy, date);
    }

    #[test]
    fn computes_day_deltas() {
        let start = SimpleDate::parse("2026-02-10").expect("date");
        let end = SimpleDate::parse("2027-02-10").expect("date");
        assert_eq!(start.days_between(&end), 365);
    }
}
