import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epistemic import Belief, Claim, Status, Verified, Estimated, recommend_action


def main():
    b = Belief([
        Claim(100, source="api_a", status=Status.VERIFIED),
        Claim(85, source="api_b", status=Status.VERIFIED),
    ])
    print("conflicts:", b.has_unresolved_conflicts())
    print("recommend:", recommend_action(b).kind)

    combined = Verified(5, source="api") + Estimated(2)
    print("combined status:", combined.status_at(None))


if __name__ == "__main__":
    main()
