use std::collections::HashMap;

use crate::claim::Epistemic;
use crate::policies::{Claims, Policy, PolicyViolation};

pub fn wrap_tool_result<T>(source: impl Into<String>, result: T) -> Epistemic<T> {
    Epistemic::estimated(result)
        .with_source(source)
        .with_kind("tool_result")
        .with_confidence(0.7)
}

pub struct EpistemicActionGate<T> {
    policies: HashMap<String, Vec<Box<dyn Policy<T>>>>,
}

impl<T> EpistemicActionGate<T> {
    pub fn new() -> Self {
        EpistemicActionGate {
            policies: HashMap::new(),
        }
    }

    pub fn register(&mut self, action_name: impl Into<String>, policies: Vec<Box<dyn Policy<T>>>) {
        self.policies.insert(action_name.into(), policies);
    }

    pub fn policies_for(&self, action_name: &str) -> &[Box<dyn Policy<T>>] {
        self.policies
            .get(action_name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn check<C: Claims<T>>(&self, action_name: &str, value: &C) -> Result<(), PolicyViolation> {
        if let Some(policies) = self.policies.get(action_name) {
            for policy in policies {
                if !policy.satisfied(value) {
                    return Err(PolicyViolation {
                        policy_name: policy.name().to_string(),
                        message: format!("policy {} not satisfied", policy.name()),
                    });
                }
            }
        }
        Ok(())
    }
}

impl<T> Default for EpistemicActionGate<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::VerifiedBy;

    #[test]
    fn gate_check_passes_when_no_policies_registered() {
        let gate: EpistemicActionGate<u32> = EpistemicActionGate::new();
        let claim = Epistemic::verified(5u32);
        assert!(gate.check("pay", &claim).is_ok());
    }

    #[test]
    fn gate_enforces_registered_policies() {
        let mut gate: EpistemicActionGate<u32> = EpistemicActionGate::new();
        gate.register(
            "payment.execute",
            vec![Box::new(VerifiedBy::new(["finance_system"]))],
        );
        let estimated = Epistemic::estimated(5u32).with_source("finance_system");
        assert!(gate.check("payment.execute", &estimated).is_err());
        let verified = Epistemic::verified(5u32).with_source("finance_system");
        assert!(gate.check("payment.execute", &verified).is_ok());
    }

    #[test]
    fn gate_policies_for() {
        let mut gate: EpistemicActionGate<u32> = EpistemicActionGate::new();
        gate.register("a", vec![]);
        assert!(gate.policies_for("a").is_empty());
        assert!(gate.policies_for("b").is_empty());
    }
}
