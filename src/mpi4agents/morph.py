from mpi4py import MPI
from pydantic_ai.models import Model

from mpi4agents.base import LLMAgent

ROLES = {
    "editor": "Rewrite the DRAFT into a better answer using the team's NOTES.",
    "researcher": "Add missing facts, examples or details the DRAFT needs.",
    "critic": "Point out errors, weak arguments or contradictions in the DRAFT.",
    "simplifier": "Suggest how to make the DRAFT shorter and clearer.",
}


class MorphAgent(LLMAgent):
    """Role-morphing: every round each rank picks the role it thinks the shared draft needs
    most. Exactly one rank becomes the editor, who merges everyone's notes and broadcasts
    the new draft. Roles change from round to round as the draft evolves."""

    def __init__(self, comm: MPI.Comm, model: Model, rounds: int = 3):
        super().__init__(comm=comm, model=model)
        self.rounds = rounds

    def choose(self, prompt: str, draft: str) -> str:
        roles_str = "\n".join(f"- {name}: {desc}" for name, desc in ROLES.items())
        output = self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nDRAFT:\n\n{draft or '(empty)'}\n\nROLES:\n\n{roles_str}",
            "Pick the one role from ROLES that would most improve the DRAFT right now. "
            "Reply with only the role name.",
        ).lower()
        return next((name for name in ROLES if name in output), "critic")

    @staticmethod
    def assign(prefs: list[str], iround: int) -> list[str]:
        """Resolve preferences so exactly one rank is the editor. Priority rotates each round."""
        size = len(prefs)
        order = [(iround + i) % size for i in range(size)]
        editor = next((r for r in order if prefs[r] == "editor"), order[0])
        others = [name for name in ROLES if name != "editor"]
        roles = []
        for r, pref in enumerate(prefs):
            if r == editor:
                roles.append("editor")
            elif pref == "editor":
                # displaced editors take the least-used support role
                taken = roles + prefs[r + 1 :]
                roles.append(min(others, key=taken.count))
            else:
                roles.append(pref)
        return roles

    def act(self, role: str, prompt: str, draft: str) -> str:
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nDRAFT:\n\n{draft or '(empty)'}",
            f"You are the {role}. {ROLES[role]} Reply with brief notes only.",
        )

    def edit(self, prompt: str, draft: str, notes: list[tuple[int, str, str]]) -> str:
        notes_str = "\n\n".join(
            f"NOTE FROM {role.upper()} (rank {r}):\n\n{n}" for r, role, n in notes if n
        )
        return self.ask(
            f"PROMPT:\n\n\t{prompt}\n\nDRAFT:\n\n{draft or '(empty)'}\n\nNOTES:\n\n{notes_str or '(none)'}",
            f"You are the editor. {ROLES['editor']} If the DRAFT is empty, write an initial "
            "answer. Reply with the full revised answer only.",
        )

    def run(self, prompt: str) -> str:
        draft = ""
        history = []
        for iround in range(self.rounds):
            roles = self.assign(self.allgather(self.choose(prompt, draft)), iround)
            role = roles[self.rank]
            editor = roles.index("editor")
            history.append(role)

            note = "" if role == "editor" else self.act(role, prompt, draft)
            notes = self.gather((self.rank, role, note), root=editor)
            draft = self.bcast(
                self.edit(prompt, draft, notes) if self.rank == editor else None,
                root=editor,
            )

        roles_str = " → ".join(history)
        return (
            f"Roles: {roles_str}\n\n{draft}"
            if self.rank == 0
            else f"Roles: {roles_str}"
        )
