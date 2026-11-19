use std::collections::HashSet;
use std::fmt;
use std::time::{Duration, SystemTime};

use crate::belief::Belief;
use crate::claim::{Epistemic, Status};

pub trait Claims<T> {
    fn claims(&self) -> Vec<&Epistemic<T>>;
    fn has_unresolved_conflicts(&self) -> bool;
}

impl<T> Claims<T> for Epistemic<T> {
    fn claims(&self) -> Vec<&Epistemic<T>> {
        vec![self]
    }

    fn has_unresolved_conflicts(&self) -> bool {
        false
    }
}

impl<T: PartialEq> Claims<T> for Belief<T> {
    fn claims(&self) -> Vec<&Epistemic<T>> {
        self.claims.iter().collect()
    }

    fn has_unresolved_conflicts(&self) -> bool {
        Belief::has_unresolved_conflicts(self)
    }
}

impl<T> Claims<T> for crate::claim::Verified<T> {
    fn claims(&self) -> Vec<&Epistemic<T>> {
        vec![self.as_epistemic()]
    }

    fn has_unresolved_conflicts(&self) -> bool {
        false
    }
}

pub trait Policy<T> {
    fn name(&self) -> &'static str;
    fn satisfied(&self, value: &dyn Claims<T>) -> bool;
}

#[derive(Debug, Clone)]
pub struct PolicyViolation {
    pub policy_name: String,
    pub message: String,
}

impl fmt::Display for PolicyViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "policy violation [{}]: {}", self.policy_name, self.message)
    }
}

impl std::error::Error for PolicyViolation {}

#[derive(Debug, Clone)]
pub struct VerifiedBy {
    pub sources: Vec<String>,
    pub min_independent: usize,
}

impl VerifiedBy {
    pub fn new(sources: impl IntoIterator<Item = impl Into<String>>) -> Self {
        VerifiedBy {
            sources: sources.into_iter().map(Into::into).collect(),
            min_independent: 1,
        }
    }

    pub fn with_min_independent(mut self, min: usize) -> Self {
        self.min_independent = min;
        self
    }
}

impl<T> Policy<T> for VerifiedBy {
    fn name(&self) -> &'static str {
        "verified_by"
    }

    fn satisfied(&self, value: &dyn Claims<T>) -> bool {
        let mut distinct: HashSet<String> = HashSet::new();
        for claim in value.claims() {
            if claim.status_at(None) == Status::Verified {
                if let Some(src) = &claim.source {
                    if self.sources.iter().any(|s| s == src) {
                        distinct.insert(src.clone());
                    }
                }
            }
        }
        distinct.len() >= self.min_independent
    }
}

#[derive(Debug, Clone)]
pub struct IndependentConfirmation {
    pub sources: Option<Vec<String>>,
    pub min_sources: usize,
}

impl Default for IndependentConfirmation {
    fn default() -> Self {
        IndependentConfirmation {
            sources: None,
            min_sources: 2,
        }
    }
}

impl IndependentConfirmation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sources(mut self, sources: Vec<String>) -> Self {
        self.sources = Some(sources);
        self
    }

    pub fn with_min_sources(mut self, min: usize) -> Self {
        self.min_sources = min;
        self
    }
}

impl<T> Policy<T> for IndependentConfirmation {
    fn name(&self) -> &'static str {
        "independent_confirmation"
    }

    fn satisfied(&self, value: &dyn Claims<T>) -> bool {
        let mut distinct: HashSet<String> = HashSet::new();
        for claim in value.claims() {
            if claim.status_at(None) == Status::Verified {
                if let Some(src) = &claim.source {
                    let allowed = self
                        .sources
                        .as_ref()
                        .map(|sources| sources.iter().any(|s| s == src))
                        .unwrap_or(true);
                    if allowed {
                        distinct.insert(src.clone());
                    }
                }
            }
        }
        distinct.len() >= self.min_sources
    }
}

#[derive(Debug, Clone)]
pub struct Fresh {
    pub within: Duration,
}

impl Fresh {
    pub fn new(within: Duration) -> Self {
        Fresh { within }
    }
}

impl<T> Policy<T> for Fresh {
    fn name(&self) -> &'static str {
        "fresh"
    }

    fn satisfied(&self, value: &dyn Claims<T>) -> bool {
        let claims = value.claims();
        let best = claims
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(Ordering::Equal));
        match best {
            Some(claim) => match claim.expires_at {
                Some(expires) => {
                    expires
                        .duration_since(SystemTime::now())
                        .map(|d| d >= self.within)
                        .unwrap_or(false)
                }
                None => false,
            },
            None => false,
        }
    }
}

use std::cmp::Ordering;

#[derive(Debug, Clone, Default)]
pub struct NoUnresolvedConflicts;

impl<T: PartialEq> Policy<T> for NoUnresolvedConflicts {
    fn name(&self) -> &'static str {
        "no_unresolved_conflicts"
    }

    fn satisfied(&self, value: &dyn Claims<T>) -> bool {
        let claims = value.claims();
        if claims.is_empty() {
            return true;
        }
        if claims.len() == 1 {
            let claim = claims[0];
            return claim.alternatives.is_empty() && claim.status != Status::Conflicting;
        }
        let first = claims[0].value();
        !claims[1..]
            .iter()
            .any(|c| c.status == Status::Conflicting || c.value() != first)
    }
}

pub fn enforce<T>(
    value: &dyn Claims<T>,
    policies: &[&dyn Policy<T>],
) -> Result<(), PolicyViolation> {
    for policy in policies {
        if !policy.satisfied(value) {
            return Err(PolicyViolation {
                policy_name: policy.name().to_string(),
                message: format!("policy {} not satisfied", policy.name()),
            });
        }
    }
    Ok(())
}

pub fn check_sensitive<'a, T>(
    value: &'a Epistemic<T>,
    allowed_statuses: &[Status],
    min_confidence: f64,
) -> Result<&'a T, PolicyViolation> {
    let status = value.status_at(None);
    if !allowed_statuses.contains(&status) {
        return Err(PolicyViolation {
            policy_name: "check_sensitive".to_string(),
            message: format!("status {status:?} not allowed"),
        });
    }
    if value.confidence < min_confidence {
        return Err(PolicyViolation {
            policy_name: "check_sensitive".to_string(),
            message: format!(
                "confidence {} below threshold {min_confidence}",
                value.confidence
            ),
        });
    }
    value
        .value()
        .ok_or_else(|| PolicyViolation {
            policy_name: "check_sensitive".to_string(),
            message: "value is missing".to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::Belief;
    use crate::middleware::wrap_tool_result;

    #[test]
    fn verified_by_passes_for_verified_claim_from_source() {
        let claim = Epistemic::verified(50u32).with_source("finance_system");
        assert!(VerifiedBy::new(["finance_system"]).satisfied(&claim));
    }

    #[test]
    fn verified_by_fails_for_estimated_claim() {
        let claim = Epistemic::estimated(50u32).with_source("finance_system");
        assert!(!VerifiedBy::new(["finance_system"]).satisfied(&claim));
    }

    #[test]
    fn verified_by_fails_for_other_source() {
        let claim = Epistemic::verified(50u32).with_source("other");
        assert!(!VerifiedBy::new(["finance_system"]).satisfied(&claim));
    }

    #[test]
    fn verified_by_on_belief_distinct_sources() {
        let belief = Belief::new(vec![
            Epistemic::verified(100u32).with_source("api_a"),
            Epistemic::verified(100u32).with_source("api_b"),
        ]);
        assert!(VerifiedBy::new(["api_a", "api_b"])
            .with_min_independent(2)
            .satisfied(&belief));
    }

    #[test]
    fn independent_confirmation_requires_distinct_sources() {
        let belief = Belief::new(vec![
            Epistemic::verified(100u32).with_source("api_a"),
            Epistemic::verified(100u32).with_source("api_a"),
            Epistemic::verified(100u32).with_source("api_a"),
        ]);
        let policy = IndependentConfirmation::new()
            .with_sources(vec!["api_a".to_string()])
            .with_min_sources(2);
        assert!(!policy.satisfied(&belief));
    }

    #[test]
    fn independent_confirmation_passes_with_distinct_sources() {
        let belief = Belief::new(vec![
            Epistemic::verified(100u32).with_source("api_a"),
            Epistemic::verified(100u32).with_source("api_b"),
        ]);
        assert!(IndependentConfirmation::new()
            .with_min_sources(2)
            .satisfied(&belief));
    }

    #[test]
    fn fresh_requires_expiry_window() {
        let now = SystemTime::now();
        let ok = Epistemic::verified(1u32).with_expires_at(now + Duration::from_secs(7200));
        assert!(Fresh::new(Duration::from_secs(3600)).satisfied(&ok));
        let too_stale = Epistemic::verified(1u32).with_expires_at(now + Duration::from_secs(7200));
        assert!(!Fresh::new(Duration::from_secs(86400)).satisfied(&too_stale));
        let no_expiry = Epistemic::verified(1u32);
        assert!(!Fresh::new(Duration::from_secs(3600)).satisfied(&no_expiry));
    }

    #[test]
    fn no_unresolved_conflicts() {
        assert!(NoUnresolvedConflicts.satisfied(&Epistemic::verified(1u32)));
        let belief = Belief::new(vec![
            Epistemic::verified(1u32).with_source("a"),
            Epistemic::verified(2u32).with_source("b"),
        ]);
        assert!(!NoUnresolvedConflicts.satisfied(&belief));
    }

    #[test]
    fn enforce_passes_and_raises() {
        let claim = Epistemic::verified(5u32).with_source("finance_system");
        let policy: &dyn Policy<u32> = &VerifiedBy::new(["finance_system"]);
        assert!(enforce(&claim, &[policy]).is_ok());
        let estimated = Epistemic::estimated(5u32).with_source("finance_system");
        assert!(enforce(&estimated, &[policy]).is_err());
    }

    #[test]
    fn enforce_on_belief_with_conflicts() {
        let belief = Belief::new(vec![
            Epistemic::verified(1u32).with_source("a"),
            Epistemic::verified(2u32).with_source("b"),
        ]);
        let policy: &dyn Policy<u32> = &NoUnresolvedConflicts;
        assert!(enforce(&belief, &[policy]).is_err());
    }

    #[test]
    fn check_sensitive_verified_passes() {
        let v = Epistemic::verified(5u32);
        assert_eq!(check_sensitive(&v, &[Status::Verified], 0.95).unwrap(), &5);
    }

    #[test]
    fn check_sensitive_estimated_raises() {
        let v = Epistemic::estimated(5u32);
        assert!(check_sensitive(&v, &[Status::Verified], 0.95).is_err());
    }

    #[test]
    fn check_sensitive_threshold_configurable() {
        let v = Epistemic::verified(5u32).with_confidence(0.9);
        assert_eq!(check_sensitive(&v, &[Status::Verified], 0.9).unwrap(), &5);
    }

    #[test]
    fn check_sensitive_allowed_statuses() {
        let v = Epistemic::stale(5u32).with_confidence(0.99);
        assert!(check_sensitive(&v, &[Status::Stale], 0.95).is_ok());
        assert!(check_sensitive(&v, &[Status::Verified], 0.95).is_err());
    }

    #[test]
    fn wrap_tool_result_helper() {
        let claim = wrap_tool_result("search_api", "result");
        assert_eq!(claim.value().copied(), Some("result"));
        assert_eq!(claim.source.as_deref(), Some("search_api"));
        assert_eq!(claim.kind.as_deref(), Some("tool_result"));
    }
}
