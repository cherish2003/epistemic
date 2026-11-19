use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::SystemTime;

use crate::claim::{Epistemic, Status};
use crate::policies::{Claims, Policy, enforce};

#[derive(Clone, Debug)]
pub struct Belief<T> {
    pub claims: Vec<Epistemic<T>>,
}

impl<T> Belief<T> {
    pub fn new(claims: Vec<Epistemic<T>>) -> Self {
        Belief { claims }
    }

    pub fn has_unresolved_conflicts(&self) -> bool
    where
        T: PartialEq,
    {
        if self.claims.len() < 2 {
            return false;
        }
        let first = self.claims[0].value();
        self.claims[1..]
            .iter()
            .any(|c| c.status == Status::Conflicting || c.value() != first)
    }

    pub fn values(&self) -> Vec<Option<&T>> {
        self.claims.iter().map(|c| c.value()).collect()
    }

    pub fn sources(&self) -> Vec<String> {
        self.claims
            .iter()
            .filter_map(|c| c.source.clone())
            .collect()
    }

    pub fn candidates(&self) -> &[Epistemic<T>] {
        &self.claims
    }

    pub fn best(&self) -> Option<&Epistemic<T>> {
        self.claims
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(Ordering::Equal))
    }

    pub fn best_matching(&self, policies: &[&dyn Policy<T>]) -> Option<&Epistemic<T>>
    where
        T: Clone + PartialEq,
    {
        self.claims
            .iter()
            .filter(|c| {
                let single = Belief::new(vec![(*c).clone()]);
                policies.iter().all(|p| p.satisfied(&single))
            })
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(Ordering::Equal))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationKind {
    Verify,
    Revalidate,
    Observe,
    AskHuman,
    None,
}

#[derive(Debug, Clone)]
pub struct Recommendation<T> {
    pub kind: RecommendationKind,
    pub claim: Epistemic<T>,
    pub rationale: String,
    pub options: Vec<String>,
}

pub fn recommend_action<T: Clone>(
    value: &dyn Claims<T>,
    policies: &[&dyn Policy<T>],
    now: Option<SystemTime>,
) -> Recommendation<T> {
    let claims = value.claims();
    if claims.is_empty() {
        return Recommendation {
            kind: RecommendationKind::AskHuman,
            claim: Epistemic::unknown(),
            rationale: "no claims to reason over".to_string(),
            options: Vec::new(),
        };
    }
    let best: Epistemic<T> = claims
        .iter()
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(Ordering::Equal))
        .copied()
        .expect("claims is non-empty")
        .clone();
    if value.has_unresolved_conflicts() {
        let verified: HashSet<String> = claims
            .iter()
            .filter(|c| c.status_at(None) == Status::Verified)
            .filter_map(|c| c.source.clone())
            .collect();
        let options: Vec<String> = claims
            .iter()
            .filter_map(|c| c.source.clone())
            .filter(|s| !verified.contains(s))
            .collect();
        return Recommendation {
            kind: RecommendationKind::Verify,
            claim: best,
            rationale: "belief has unresolved conflicts; resolve via independent sources"
                .to_string(),
            options,
        };
    }
    if claims.iter().any(|c| c.status_at(now) == Status::Stale) {
        return Recommendation {
            kind: RecommendationKind::Revalidate,
            claim: best,
            rationale: "the claim is stale; re-fetch evidence".to_string(),
            options: Vec::new(),
        };
    }
    if claims.iter().any(|c| c.status == Status::Unknown) {
        let options: Vec<String> = claims
            .iter()
            .filter_map(|c| c.source.clone())
            .collect();
        return Recommendation {
            kind: RecommendationKind::Observe,
            claim: best,
            rationale: "the value is unknown; observe the listed sources".to_string(),
            options,
        };
    }
    if enforce(value, policies).is_err() {
        return Recommendation {
            kind: RecommendationKind::AskHuman,
            claim: best,
            rationale: "policies block the value and no further evidence is available".to_string(),
            options: Vec::new(),
        };
    }
    Recommendation {
        kind: RecommendationKind::None,
        claim: best,
        rationale: "no action needed".to_string(),
        options: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn belief_conflicting_values() {
        let b = Belief::new(vec![
            Epistemic::verified(100u32),
            Epistemic::verified(85u32),
            Epistemic::verified(120u32),
        ]);
        assert!(b.has_unresolved_conflicts());
    }

    #[test]
    fn belief_no_conflict_when_consistent() {
        assert!(!Belief::new(vec![Epistemic::verified(100u32)]).has_unresolved_conflicts());
        assert!(!Belief::new(vec![
            Epistemic::verified(100u32),
            Epistemic::verified(100u32),
        ])
        .has_unresolved_conflicts());
    }

    #[test]
    fn belief_best_by_confidence() {
        let b = Belief::new(vec![
            Epistemic::estimated(5u32).with_confidence(0.9),
            Epistemic::estimated(7u32).with_confidence(0.2),
        ]);
        assert_eq!(b.best().unwrap().value().copied(), Some(5));
    }

    #[test]
    fn belief_best_matching() {
        use crate::policies::{Claims, Policy};

        struct OnlyVerified;

        impl<T> Policy<T> for OnlyVerified {
            fn name(&self) -> &'static str {
                "only_verified"
            }
            fn satisfied(&self, value: &dyn Claims<T>) -> bool {
                value.claims().iter().all(|c| c.status == Status::Verified)
            }
        }

        let b = Belief::new(vec![
            Epistemic::verified(5u32).with_confidence(0.2),
            Epistemic::verified(7u32).with_confidence(0.9),
        ]);
        let policies: [&dyn Policy<u32>; 1] = [&OnlyVerified];
        let best = b.best_matching(&policies);
        assert_eq!(best.unwrap().value().copied(), Some(7));
    }

    #[test]
    fn belief_sources_and_values() {
        let b = Belief::new(vec![
            Epistemic::verified(1u32).with_source("a"),
            Epistemic::verified(2u32).with_source("b"),
        ]);
        let mut sources = b.sources();
        sources.sort();
        assert_eq!(sources, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(b.values(), vec![Some(&1), Some(&2)]);
    }

    #[test]
    fn belief_candidates() {
        let b = Belief::new(vec![Epistemic::verified(1u32), Epistemic::verified(2u32)]);
        assert_eq!(b.candidates().len(), 2);
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use crate::policies::VerifiedBy;
    use std::time::Duration;

    #[test]
    fn recommend_none_when_satisfied() {
        let b = Belief::new(vec![Epistemic::verified(5u32).with_source("api")]);
        let r = recommend_action(&b, &[], None);
        assert_eq!(r.kind, RecommendationKind::None);
    }

    #[test]
    fn recommend_observe_when_unknown() {
        let b: Belief<u32> = Belief::new(vec![Epistemic::unknown().with_source("search_api")]);
        let r = recommend_action(&b, &[], None);
        assert_eq!(r.kind, RecommendationKind::Observe);
        assert!(r.options.contains(&"search_api".to_string()));
    }

    #[test]
    fn recommend_revalidate_when_stale() {
        let b = Belief::new(vec![Epistemic::verified(5u32)
            .with_expires_at(SystemTime::now() - Duration::from_secs(1))]);
        let r = recommend_action(&b, &[], None);
        assert_eq!(r.kind, RecommendationKind::Revalidate);
    }

    #[test]
    fn recommend_verify_on_unresolved_conflicts() {
        let b = Belief::new(vec![
            Epistemic::verified(100u32).with_source("api_a"),
            Epistemic::verified(85u32).with_source("api_b"),
        ]);
        let r = recommend_action(&b, &[], None);
        assert_eq!(r.kind, RecommendationKind::Verify);
    }

    #[test]
    fn recommend_ask_human_when_blocked() {
        let b = Belief::new(vec![Epistemic::estimated(5u32).with_source("finance")]);
        let policy: &dyn Policy<u32> = &VerifiedBy::new(["finance_system"]);
        let r = recommend_action(&b, &[policy], None);
        assert_eq!(r.kind, RecommendationKind::AskHuman);
    }

    #[test]
    fn recommend_action_on_single_epistemic() {
        let v: Epistemic<u32> = Epistemic::unknown();
        let r = recommend_action(&v, &[], None);
        assert_eq!(r.kind, RecommendationKind::Observe);
    }
}
