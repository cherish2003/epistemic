pub mod belief;
pub mod claim;
pub mod middleware;
pub mod policies;

pub use belief::{Belief, Recommendation, RecommendationKind, recommend_action};
pub use claim::{combine, Derivation, Epistemic, Evidence, Status, Verified};
pub use middleware::{wrap_tool_result, EpistemicActionGate};
pub use policies::{
    check_sensitive, enforce, Claims, Fresh, IndependentConfirmation, NoUnresolvedConflicts,
    Policy, PolicyViolation, VerifiedBy,
};
