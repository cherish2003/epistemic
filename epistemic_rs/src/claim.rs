use std::ops::Add;
use std::time::SystemTime;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    Verified,
    Stale,
    Estimated,
    Unknown,
    Conflicting,
}

impl Status {
    pub fn default_confidence(&self) -> f64 {
        match self {
            Status::Verified => 1.0,
            Status::Stale => 0.8,
            Status::Estimated => 0.5,
            Status::Unknown => 0.0,
            Status::Conflicting => 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evidence {
    pub source: String,
    pub kind: String,
    pub timestamp: Option<SystemTime>,
    pub description: Option<String>,
}

impl Evidence {
    pub fn new(source: impl Into<String>) -> Self {
        Evidence {
            source: source.into(),
            kind: "observation".to_string(),
            timestamp: None,
            description: None,
        }
    }

    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = kind.into();
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

#[derive(Clone, Debug)]
pub struct Epistemic<T> {
    value: Option<T>,
    pub status: Status,
    pub confidence: f64,
    pub source: Option<String>,
    pub expires_at: Option<SystemTime>,
    pub evidence: Vec<Evidence>,
    pub notes: Option<String>,
    pub kind: Option<String>,
    pub observed_at: SystemTime,
    pub alternatives: Vec<Epistemic<T>>,
}

impl<T> Epistemic<T> {
    pub fn new(status: Status, value: Option<T>) -> Self {
        Epistemic {
            value,
            status,
            confidence: status.default_confidence(),
            source: None,
            expires_at: None,
            evidence: Vec::new(),
            notes: None,
            kind: None,
            observed_at: SystemTime::now(),
            alternatives: Vec::new(),
        }
    }

    pub fn verified(value: T) -> Self {
        Self::new(Status::Verified, Some(value))
    }

    pub fn stale(value: T) -> Self {
        Self::new(Status::Stale, Some(value))
    }

    pub fn estimated(value: T) -> Self {
        Self::new(Status::Estimated, Some(value))
    }

    pub fn unknown() -> Self {
        Self::new(Status::Unknown, None)
    }

    pub fn conflicting(alternatives: Vec<Epistemic<T>>) -> Self {
        let mut claim = Self::new(Status::Conflicting, None);
        claim.alternatives = alternatives;
        claim
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_expires_at(mut self, expires_at: SystemTime) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    pub fn with_evidence(mut self, evidence: Vec<Evidence>) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = Some(kind.into());
        self
    }

    pub fn with_alternatives(mut self, alternatives: Vec<Epistemic<T>>) -> Self {
        self.alternatives = alternatives;
        self
    }

    pub fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    pub fn as_value(&self) -> Option<&T> {
        self.value()
    }

    pub fn is_expired(&self, now: Option<SystemTime>) -> bool {
        match self.expires_at {
            Some(expires) => now.unwrap_or_else(SystemTime::now) >= expires,
            None => false,
        }
    }

    pub fn is_current(&self, now: Option<SystemTime>) -> bool {
        !self.is_expired(now)
    }

    pub fn status_at(&self, now: Option<SystemTime>) -> Status {
        if self.is_expired(now)
            && matches!(
                self.status,
                Status::Verified | Status::Stale | Status::Estimated
            )
        {
            Status::Stale
        } else {
            self.status
        }
    }

    pub fn verify(self) -> Option<Verified<T>> {
        if self.status_at(None) == Status::Verified {
            Some(Verified(self))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct Verified<T>(Epistemic<T>);

impl<T> Verified<T> {
    pub fn value(&self) -> &T {
        self.0.value.as_ref().expect("verified claims carry a value")
    }

    pub fn confidence(&self) -> f64 {
        self.0.confidence
    }

    pub fn source(&self) -> Option<&str> {
        self.0.source.as_deref()
    }

    pub fn as_epistemic(&self) -> &Epistemic<T> {
        &self.0
    }
}

pub struct Derivation<T> {
    pub value: T,
    pub status: Status,
    pub confidence: f64,
    pub evidence: Vec<Evidence>,
    pub inputs: Vec<Epistemic<T>>,
    pub op: String,
}

impl<T> Derivation<T> {
    pub fn from_inputs(value: T, op: impl Into<String>, inputs: Vec<Epistemic<T>>) -> Self {
        let status = inputs
            .iter()
            .map(|i| i.status_at(None))
            .max()
            .unwrap_or(Status::Verified);
        let confidence = if inputs.is_empty() {
            status.default_confidence()
        } else {
            inputs
                .iter()
                .map(|i| i.confidence)
                .fold(f64::MAX, f64::min)
        };
        Derivation {
            value,
            status,
            confidence,
            evidence: inputs
                .iter()
                .flat_map(|i| i.evidence.iter().cloned())
                .collect(),
            inputs,
            op: op.into(),
        }
    }
}

fn flatten_conflicts<T: Clone>(values: &[Epistemic<T>]) -> Vec<Epistemic<T>> {
    let mut out = Vec::new();
    for v in values {
        if v.status == Status::Conflicting && !v.alternatives.is_empty() {
            out.extend(flatten_conflicts(&v.alternatives));
        } else {
            out.push(v.clone());
        }
    }
    out
}

pub fn combine<T: Clone>(
    values: &[Epistemic<T>],
    op: impl FnOnce(Vec<T>) -> T,
) -> Epistemic<T> {
    assert!(!values.is_empty(), "combine requires at least one value");
    let statuses: Vec<Status> = values.iter().map(|v| v.status_at(None)).collect();
    let worst = statuses.iter().max().copied().unwrap_or(Status::Verified);
    if worst == Status::Conflicting {
        return Epistemic::conflicting(flatten_conflicts(values));
    }
    if worst == Status::Unknown {
        return Epistemic::unknown();
    }
    let present: Vec<T> = values.iter().filter_map(|v| v.value().cloned()).collect();
    let value = if present.len() == values.len() {
        Some(op(present))
    } else {
        None
    };
    let confidence = values
        .iter()
        .map(|v| v.confidence)
        .fold(f64::MAX, f64::min);
    let mut combined = Epistemic::new(worst, value);
    combined.confidence = confidence;
    combined.evidence = values
        .iter()
        .flat_map(|v| v.evidence.iter().cloned())
        .collect();
    let notes: Vec<String> = values.iter().filter_map(|v| v.notes.clone()).collect();
    if !notes.is_empty() {
        combined.notes = Some(notes.join("; "));
    }
    combined
}

impl<T: Add<Output = T> + Clone> Add for Epistemic<T> {
    type Output = Epistemic<T>;

    fn add(self, rhs: Epistemic<T>) -> Epistemic<T> {
        combine(&[self, rhs], |values: Vec<T>| {
            let mut it = values.into_iter();
            let first = it.next().expect("combine requires at least one value");
            it.fold(first, |acc, v| acc + v)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn precedence_order() {
        assert!(Status::Verified < Status::Stale);
        assert!(Status::Stale < Status::Estimated);
        assert!(Status::Estimated < Status::Unknown);
        assert!(Status::Unknown < Status::Conflicting);
    }

    #[test]
    fn verified_defaults() {
        let v = Epistemic::verified(500u32);
        assert_eq!(v.value().copied(), Some(500));
        assert_eq!(v.status, Status::Verified);
        assert_eq!(v.confidence, 1.0);
        assert_eq!(v.source, None);
        assert!(v.evidence.is_empty());
        assert!(v.alternatives.is_empty());
    }

    #[test]
    fn stale_defaults() {
        let v = Epistemic::stale(5u32);
        assert_eq!(v.status, Status::Stale);
        assert_eq!(v.confidence, 0.8);
    }

    #[test]
    fn estimated_defaults() {
        let v = Epistemic::estimated(0.72f64);
        assert_eq!(v.status, Status::Estimated);
        assert_eq!(v.confidence, 0.5);
    }

    #[test]
    fn unknown_defaults() {
        let v = Epistemic::<u32>::unknown();
        assert_eq!(v.status, Status::Unknown);
        assert_eq!(v.value(), None);
        assert_eq!(v.confidence, 0.0);
    }

    #[test]
    fn conflicting_defaults() {
        let v = Epistemic::conflicting(vec![
            Epistemic::verified(1u32),
            Epistemic::estimated(2u32),
        ]);
        assert_eq!(v.status, Status::Conflicting);
        assert_eq!(v.alternatives.len(), 2);
    }

    #[test]
    fn expired_is_stale() {
        let v =
            Epistemic::verified(5u32).with_expires_at(SystemTime::now() - Duration::from_secs(1));
        assert!(v.is_expired(None));
        assert_eq!(v.status_at(None), Status::Stale);
    }

    #[test]
    fn not_expired() {
        let v = Epistemic::verified(5u32)
            .with_expires_at(SystemTime::now() + Duration::from_secs(3600));
        assert!(!v.is_expired(None));
        assert_eq!(v.status_at(None), Status::Verified);
    }

    #[test]
    fn no_expiry() {
        assert!(!Epistemic::verified(5u32).is_expired(None));
    }

    #[test]
    fn evidence_attachment() {
        let e = Evidence::new("api").with_kind("observation").with_description("obs");
        let v = Epistemic::verified(1u32).with_evidence(vec![e.clone()]);
        assert_eq!(v.evidence.len(), 1);
        assert_eq!(v.evidence[0].source, "api");
        assert_eq!(v.evidence[0].kind, "observation");
    }

    #[test]
    fn combine_verified_plus_verified() {
        let v = combine(
            &[Epistemic::verified(2u32), Epistemic::verified(3u32)],
            |xs| xs.into_iter().sum(),
        );
        assert_eq!(v.value().copied(), Some(5));
        assert_eq!(v.status, Status::Verified);
        assert_eq!(v.confidence, 1.0);
    }

    #[test]
    fn combine_verified_plus_estimated_propagates() {
        let v = combine(
            &[Epistemic::verified(2u32), Epistemic::estimated(3u32)],
            |xs| xs.into_iter().sum(),
        );
        assert_eq!(v.value().copied(), Some(5));
        assert_eq!(v.status, Status::Estimated);
    }

    #[test]
    fn combine_stale_propagates() {
        let v = combine(
            &[Epistemic::stale(1u32), Epistemic::verified(2u32)],
            |xs| xs.into_iter().sum(),
        );
        assert_eq!(v.status, Status::Stale);
        assert_eq!(v.value().copied(), Some(3));
    }

    #[test]
    fn combine_expired_propagates_as_stale() {
        let expired = Epistemic::verified(1u32)
            .with_expires_at(SystemTime::now() - Duration::from_secs(1));
        let v = combine(&[expired, Epistemic::verified(2u32)], |xs| xs.into_iter().sum());
        assert_eq!(v.status, Status::Stale);
    }

    #[test]
    fn combine_unknown_propagates() {
        let v = combine(
            &[Epistemic::verified(1u32), Epistemic::<u32>::unknown()],
            |xs| xs.into_iter().sum(),
        );
        assert_eq!(v.status, Status::Unknown);
    }

    #[test]
    fn combine_conflicting_flattens() {
        let a = Epistemic::verified(1u32);
        let b = Epistemic::verified(2u32);
        let v = combine(
            &[
                Epistemic::verified(0u32),
                Epistemic::conflicting(vec![a, b]),
            ],
            |xs| xs.into_iter().sum(),
        );
        assert_eq!(v.status, Status::Conflicting);
        assert_eq!(v.alternatives.len(), 3);
        assert_eq!(v.alternatives[0].value().copied(), Some(0));
        assert_eq!(v.alternatives[1].value().copied(), Some(1));
        assert_eq!(v.alternatives[2].value().copied(), Some(2));
    }

    #[test]
    fn combine_confidence_min() {
        let v = combine(
            &[
                Epistemic::estimated(1u32).with_confidence(0.6),
                Epistemic::estimated(2u32).with_confidence(0.8),
            ],
            |xs| xs.into_iter().sum(),
        );
        assert_eq!(v.status, Status::Estimated);
        assert!((v.confidence - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn combine_operator_add() {
        let v = Epistemic::verified(2u32) + Epistemic::verified(3u32);
        assert_eq!(v.value().copied(), Some(5));
        assert_eq!(v.status, Status::Verified);
    }

    #[test]
    fn combine_operator_add_with_estimated() {
        let v = Epistemic::verified(2u32) + Epistemic::estimated(3u32);
        assert_eq!(v.status, Status::Estimated);
    }

    #[test]
    fn verify_returns_only_when_verified() {
        assert!(Epistemic::verified(5u32).verify().is_some());
        assert_eq!(Epistemic::verified(5u32).verify().unwrap().value(), &5);
        assert!(Epistemic::estimated(5u32).verify().is_none());
        assert!(Epistemic::stale(5u32).verify().is_none());
    }

    #[test]
    fn verify_rejects_expired() {
        let e = Epistemic::verified(5u32)
            .with_expires_at(SystemTime::now() - Duration::from_secs(1));
        assert!(e.verify().is_none());
    }

    #[test]
    fn wrap_tool_result_attaches_source_and_kind() {
        let claim = crate::middleware::wrap_tool_result("weather_api", 28u32);
        assert_eq!(claim.value().copied(), Some(28));
        assert_eq!(claim.source.as_deref(), Some("weather_api"));
        assert_eq!(claim.kind.as_deref(), Some("tool_result"));
        assert_eq!(claim.status, Status::Estimated);
    }

    #[test]
    fn combine_merges_evidence() {
        let a = Epistemic::verified(1u32).with_evidence(vec![Evidence::new("a")]);
        let b = Epistemic::verified(2u32).with_evidence(vec![Evidence::new("b")]);
        let v = combine(&[a, b], |xs| xs.into_iter().sum());
        let sources: Vec<&str> = v.evidence.iter().map(|e| e.source.as_str()).collect();
        assert_eq!(sources, vec!["a", "b"]);
    }

    #[test]
    fn derivation_propagates_status() {
        let d = Derivation::from_inputs(
            5u32,
            "add",
            vec![Epistemic::verified(2u32), Epistemic::verified(3u32)],
        );
        assert_eq!(d.value, 5);
        assert_eq!(d.status, Status::Verified);
        assert_eq!(d.inputs.len(), 2);
        assert_eq!(d.op, "add");
    }

    #[test]
    fn derivation_from_estimated_inputs() {
        let d = Derivation::from_inputs(
            3u32,
            "add",
            vec![Epistemic::estimated(1u32), Epistemic::estimated(2u32)],
        );
        assert_eq!(d.status, Status::Estimated);
        assert_eq!(d.confidence, 0.5);
    }
}
