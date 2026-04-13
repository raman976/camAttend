import json
import operator
import os
import re
from datetime import datetime
from typing import Annotated, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class AttendanceState(TypedDict):
    # ── inputs ────────────────────────────────────────────────────────────────
    student_name: str
    confidence_score: float
    current_time: datetime
    class_start_time: datetime
    student_history: Optional[Dict]
    attendance_records: List[Dict]
    lecture_context: Dict
    session_id: Optional[str]

    # ── computed in preprocess ─────────────────────────────────────────────
    time_offset_minutes: float
    normalized_history: Dict
    is_unknown: bool
    face_signature: Optional[str]
    image_quality: float
    previous_errors: int
    low_conf_ratio: float
    total_faces: int
    unknown_frequency: int
    uncertainty_score: float

    # ── output ─────────────────────────────────────────────────────────────
    decision: str
    action: str
    reasoning: str
    requires_review: bool
    agent_type: str
    override_option: bool
    context: Dict

    # ── trace: appended by every node (operator.add merges lists) ──────────
    trace: Annotated[List[str], operator.add]


class LangGraphAttendanceAgent:
    """LangGraph-based attendance decision agent.

    Graph flow:
        preprocess
            ↓
        batch_quality_guard  (exits early on low-quality batch)
            ↓
        unknown_freq_guard   (exits early on repeated unknown face)
            ↓
        confidence_gate      (exits early on very low confidence)
            ↓
        compute_uncertainty
            ↓
        route → rule_decision | llm_decision
            ↓
        finalize → END
    """

    HIGH_CONF = 0.85
    MED_CONF = 0.65
    LOW_CONF = 0.45

    GRACE = 5
    LATE_WIN = 15
    REVIEW_WIN = 30
    ABSENT_WIN = 45

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        raw_key = (
            api_key
            or os.getenv("groq_api")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_API")
        )
        # Strip surrounding quotes that some .env editors add (e.g. groq_api='key')
        self.api_key = raw_key.strip("'\"") if raw_key else None
        self.model = model
        self.groq_client = Groq(api_key=self.api_key) if self.api_key and Groq else None
        self.session_memory: Dict[str, Dict] = {}
        self.graph = self._build_graph()

    # ── graph construction ──────────────────────────────────────────────────

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE:
            return None

        g = StateGraph(AttendanceState)

        g.add_node("preprocess", self._node_preprocess)
        g.add_node("batch_quality_guard", self._node_batch_quality_guard)
        g.add_node("unknown_freq_guard", self._node_unknown_freq_guard)
        g.add_node("confidence_gate", self._node_confidence_gate)
        g.add_node("compute_uncertainty", self._node_compute_uncertainty)
        g.add_node("rule_decision", self._node_rule_decision)
        g.add_node("llm_decision", self._node_llm_decision)
        g.add_node("finalize", self._node_finalize)

        g.set_entry_point("preprocess")

        g.add_conditional_edges(
            "preprocess",
            lambda s: "retake" if (
                s.get("total_faces", 0) >= 3 and s.get("low_conf_ratio", 0) >= 0.5
            ) else "continue",
            {"retake": "batch_quality_guard", "continue": "unknown_freq_guard"},
        )

        g.add_edge("batch_quality_guard", "finalize")

        g.add_conditional_edges(
            "unknown_freq_guard",
            lambda s: "enroll" if (
                s.get("is_unknown") and s.get("unknown_frequency", 0) >= 3
            ) else "continue",
            {"enroll": "finalize", "continue": "confidence_gate"},
        )

        g.add_conditional_edges(
            "confidence_gate",
            lambda s: "escalate" if s.get("confidence_score", 0) < self.LOW_CONF else "continue",
            {"escalate": "finalize", "continue": "compute_uncertainty"},
        )

        g.add_conditional_edges(
            "compute_uncertainty",
            lambda s: (
                "llm"
                if (
                    self.MED_CONF <= s.get("confidence_score", 0) < self.HIGH_CONF
                    and self.groq_client is not None
                )
                else "rule"
            ),
            {"llm": "llm_decision", "rule": "rule_decision"},
        )

        g.add_edge("rule_decision", "finalize")
        g.add_edge("llm_decision", "finalize")
        g.add_edge("finalize", END)

        return g.compile()

    # ── nodes ───────────────────────────────────────────────────────────────

    def _node_preprocess(self, state: AttendanceState) -> dict:
        raw = state.get("student_history") or {}
        normalized = {
            "total_classes": int(raw.get("total_classes", 0)),
            "present": int(raw.get("present", 0)),
            "late": int(raw.get("late", 0)),
            "absent": int(raw.get("absent", 0)),
            "avg_attendance": float(raw.get("avg_attendance", 0.0)),
            "recent_pattern": raw.get("recent_pattern", []),
        }

        time_offset = (
            state["current_time"] - state["class_start_time"]
        ).total_seconds() / 60.0

        ctx = state.get("lecture_context", {})

        return {
            "normalized_history": normalized,
            "time_offset_minutes": time_offset,
            "is_unknown": bool(ctx.get("is_unknown", False)),
            "face_signature": ctx.get("face_signature"),
            "image_quality": float(ctx.get("image_quality", 0.7)),
            "previous_errors": int(ctx.get("previous_recognition_errors", 0)),
            "low_conf_ratio": float(ctx.get("low_conf_ratio", 0.0)),
            "total_faces": int(ctx.get("total_faces", 1)),
            "unknown_frequency": 0,
            "trace": [
                f"preprocess → student={state['student_name']} "
                f"conf={state['confidence_score']:.3f} "
                f"offset={time_offset:.1f}min"
            ],
        }

    def _node_batch_quality_guard(self, state: AttendanceState) -> dict:
        lc = state.get("low_conf_ratio", 0)
        tf = state.get("total_faces", 0)
        return {
            "decision": "FLAGGED",
            "action": "RETAKE_PHOTO",
            "reasoning": (
                f"{tf} faces detected with {lc:.0%} low-confidence detections. "
                "Photo quality is insufficient — please retake."
            ),
            "requires_review": True,
            "agent_type": "batch_quality_guard",
            "override_option": True,
            "trace": [f"batch_quality_guard → {tf} faces, {lc:.0%} low-conf → RETAKE_PHOTO"],
        }

    def _node_unknown_freq_guard(self, state: AttendanceState) -> dict:
        if not state.get("is_unknown"):
            return {"trace": ["unknown_freq_guard → known student, skip"]}

        session_id = state.get("session_id")
        sig = state.get("face_signature")
        freq = self._get_unknown_frequency(session_id, sig)

        if freq >= 3:
            return {
                "decision": "FLAGGED",
                "action": "SUGGEST_ENROLL_NEW_STUDENT",
                "reasoning": (
                    f"Unknown face appeared {freq} times this session. "
                    "Enrollment recommended."
                ),
                "requires_review": True,
                "agent_type": "unknown_frequency_guard",
                "override_option": True,
                "unknown_frequency": freq,
                "trace": [f"unknown_freq_guard → freq={freq} → SUGGEST_ENROLL"],
            }

        return {
            "unknown_frequency": freq,
            "trace": [f"unknown_freq_guard → unknown face freq={freq}, continue"],
        }

    def _node_confidence_gate(self, state: AttendanceState) -> dict:
        conf = state.get("confidence_score", 0)
        if conf < self.LOW_CONF:
            return {
                "decision": "FLAGGED",
                "action": "ESCALATE_TO_INSTRUCTOR",
                "reasoning": (
                    f"Face confidence {conf:.2%} is below the safety threshold "
                    f"({self.LOW_CONF:.0%}). Manual review required."
                ),
                "requires_review": True,
                "agent_type": "confidence_gate",
                "override_option": True,
                "trace": [
                    f"confidence_gate → conf={conf:.3f} < {self.LOW_CONF} → ESCALATE"
                ],
            }
        return {"trace": [f"confidence_gate → conf={conf:.3f} passes, continue"]}

    def _node_compute_uncertainty(self, state: AttendanceState) -> dict:
        conf = state.get("confidence_score", 0)
        history = state.get("normalized_history", {})
        quality = state.get("image_quality", 0.7)
        errors = state.get("previous_errors", 0)
        time_offset = state.get("time_offset_minutes", 0)
        low_conf_ratio = state.get("low_conf_ratio", 0)
        is_unknown = state.get("is_unknown", False)
        has_history = history.get("total_classes", 0) > 0
        attendance_rate = history.get("avg_attendance", 0.0)

        conf_risk = 1.0 - max(0.0, min(1.0, conf))
        quality_risk = 1.0 - max(0.0, min(1.0, quality))
        error_risk = min(errors / 5.0, 1.0)
        timing_risk = (
            min((time_offset - self.LATE_WIN) / 30.0, 1.0)
            if time_offset > self.LATE_WIN
            else 0.0
        )
        history_risk = (
            0.35 if not has_history else max(0.0, 0.7 - attendance_rate) / 0.7
        )
        unknown_risk = 1.0 if is_unknown else 0.0
        crowd_risk = max(0.0, min(1.0, low_conf_ratio))

        score = round(
            max(
                0.0,
                min(
                    1.0,
                    0.32 * conf_risk
                    + 0.14 * quality_risk
                    + 0.14 * error_risk
                    + 0.12 * timing_risk
                    + 0.12 * history_risk
                    + 0.10 * unknown_risk
                    + 0.06 * crowd_risk,
                ),
            ),
            3,
        )
        return {
            "uncertainty_score": score,
            "trace": [f"compute_uncertainty → score={score:.3f}"],
        }

    def _node_rule_decision(self, state: AttendanceState) -> dict:
        conf = state.get("confidence_score", 0)
        time_offset = state.get("time_offset_minutes", 0)
        history = state.get("normalized_history", {})
        uncertainty = state.get("uncertainty_score", 0)

        parts = []
        requires_review = False
        action = "MARK_PRESENT"
        decision = "PRESENT"

        if time_offset < 0:
            decision, action = "FLAGGED", "ESCALATE_TO_INSTRUCTOR"
            parts.append("Detected before class start time.")
            requires_review = True
        elif time_offset <= self.GRACE:
            decision = "PRESENT"
            parts.append(f"On time within grace period ({time_offset:.1f}min).")
        elif time_offset <= self.LATE_WIN:
            decision = "LATE"
            parts.append(f"Late window arrival ({time_offset:.1f}min after start).")
        elif time_offset <= self.REVIEW_WIN:
            if history["total_classes"] == 0:
                decision, action = "FLAGGED", "SOFT_FLAG"
                parts.append(f"No attendance history at {time_offset:.1f}min.")
                requires_review = True
            elif history.get("avg_attendance", 0) >= 0.8:
                decision, action = "LATE", "MARK_PRESENT"
                parts.append(
                    f"Strong history ({history['avg_attendance']:.0%}), "
                    f"accepting LATE at {time_offset:.1f}min."
                )
            else:
                decision, action = "FLAGGED", "ESCALATE_TO_INSTRUCTOR"
                parts.append(
                    f"Borderline timing ({time_offset:.1f}min) with weak attendance history."
                )
                requires_review = True
        else:
            decision, action = "ABSENT", "ESCALATE_TO_INSTRUCTOR"
            parts.append(f"Beyond attendance window ({time_offset:.1f}min).")

        if (
            self.MED_CONF <= conf < self.HIGH_CONF
            and uncertainty < 0.60
            and decision in {"PRESENT", "LATE"}
        ):
            requires_review = True
            action = "SOFT_FLAG"
            parts.append("Medium confidence — soft flagged for optional review.")

        if conf < self.MED_CONF and history.get("avg_attendance", 0) < 0.5:
            decision, requires_review, action = "FLAGGED", True, "ESCALATE_TO_INSTRUCTOR"
            parts.append("Low confidence combined with poor attendance history.")

        if history["total_classes"] == 0:
            parts.append("No prior attendance history — timing-only decision.")

        attendance_context = self._summarize_recent_attendance(
            state.get("attendance_records", [])
        )

        return {
            "decision": decision,
            "action": action,
            "reasoning": " ".join(parts),
            "requires_review": requires_review,
            "agent_type": "rule_based",
            "override_option": True,
            "context": {
                "student_name": state["student_name"],
                "history_available": history["total_classes"] > 0,
                "avg_attendance": history.get("avg_attendance", 0),
                "recent_pattern": history.get("recent_pattern", []),
                "attendance_context": attendance_context,
                "lecture_context": state.get("lecture_context", {}),
            },
            "trace": [f"rule_decision → {decision} / {action}"],
        }

    def _node_llm_decision(self, state: AttendanceState) -> dict:
        conf = state.get("confidence_score", 0)
        time_offset = state.get("time_offset_minutes", 0)
        history = state.get("normalized_history", {})
        attendance_records = state.get("attendance_records", [])
        lecture_context = state.get("lecture_context", {})
        recent = self._summarize_recent_attendance(attendance_records)
        has_history = history.get("total_classes", 0) > 0

        prompt = f"""You are a classroom attendance decision agent.

Make a conservative, explainable attendance decision.

Student: {state['student_name']}
Face confidence: {conf:.2%}
Minutes after class start: {time_offset:.1f}
Has history: {has_history}
Attendance rate: {history.get('avg_attendance', 0):.1%}
Present: {history.get('present', 0)}, Late: {history.get('late', 0)}, Absent: {history.get('absent', 0)}
Recent attendance: {recent}
Context: {json.dumps(lecture_context, ensure_ascii=True)}

Policy:
- PRESENT: reliable face, within early window
- LATE: reliable face, after grace period but within late window
- ABSENT: far outside attendance window
- FLAGGED: missing history, borderline confidence/timing, needs human review

Return ONLY valid JSON (no markdown, no extra text):
{{"decision": "PRESENT|LATE|ABSENT|FLAGGED", "confidence_score": 0.0, "reasoning": "...", "requires_review": true}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=300,
                stream=False,
            )
            text = response.choices[0].message.content or ""
            payload = self._parse_json(text)
            if not payload:
                raise ValueError("No valid JSON in LLM response")

            decision = payload.get("decision", "FLAGGED")
            requires_review = bool(
                payload.get("requires_review", decision == "FLAGGED")
            )
            action = (
                "SOFT_FLAG"
                if decision in {"PRESENT", "LATE"} and requires_review
                else "MARK_PRESENT"
            )

            return {
                "decision": decision,
                "action": action,
                "reasoning": payload.get("reasoning", "LLM-based decision."),
                "confidence_score": float(payload.get("confidence_score", conf)),
                "requires_review": requires_review,
                "agent_type": "llm_based",
                "override_option": True,
                "context": {
                    "student_name": state["student_name"],
                    "history_available": has_history,
                    "avg_attendance": history.get("avg_attendance", 0),
                    "recent_pattern": history.get("recent_pattern", []),
                    "attendance_context": recent,
                    "lecture_context": lecture_context,
                },
                "trace": [f"llm_decision → {decision} / {action} (Groq)"],
            }
        except Exception as exc:
            fallback = self._node_rule_decision(state)
            fallback["agent_type"] = "llm_fallback"
            fallback["reasoning"] = (
                f"LLM unavailable ({exc}). Falling back to rules. "
                + fallback.get("reasoning", "")
            )
            fallback["trace"] = [f"llm_decision → LLM failed ({exc}), rule fallback"]
            return fallback

    def _node_finalize(self, state: AttendanceState) -> dict:
        session_id = state.get("session_id")
        if session_id:
            self._record_session(session_id, state)
        return {
            "trace": [
                f"finalize → {state.get('decision','FLAGGED')} "
                f"uncertainty={state.get('uncertainty_score', 0):.3f}"
            ]
        }

    # ── public API ───────────────────────────────────────────────────────────

    def make_decision(
        self,
        student_name: str,
        confidence_score: float,
        current_time: datetime,
        class_start_time: datetime,
        student_history: Optional[Dict] = None,
        attendance_records: Optional[List[Dict]] = None,
        lecture_context: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        initial: AttendanceState = {
            "student_name": student_name,
            "confidence_score": float(confidence_score),
            "current_time": current_time,
            "class_start_time": class_start_time,
            "student_history": student_history,
            "attendance_records": attendance_records or [],
            "lecture_context": lecture_context or {},
            "session_id": session_id,
            # placeholders
            "time_offset_minutes": 0.0,
            "normalized_history": {},
            "is_unknown": False,
            "face_signature": None,
            "image_quality": 0.7,
            "previous_errors": 0,
            "low_conf_ratio": 0.0,
            "total_faces": 1,
            "unknown_frequency": 0,
            "uncertainty_score": 0.0,
            "decision": "FLAGGED",
            "action": "ESCALATE_TO_INSTRUCTOR",
            "reasoning": "",
            "requires_review": True,
            "agent_type": "unknown",
            "override_option": True,
            "context": {},
            "trace": [],
        }

        if self.graph is None:
            return self._fallback_decision(initial)

        final = self.graph.invoke(initial)

        return {
            "decision": final.get("decision", "FLAGGED"),
            "confidence": float(final.get("confidence_score", confidence_score)),
            "uncertainty_score": float(final.get("uncertainty_score", 0.0)),
            "action": final.get("action", "ESCALATE_TO_INSTRUCTOR"),
            "reasoning": final.get("reasoning", ""),
            "requires_review": bool(final.get("requires_review", True)),
            "override_option": True,
            "agent_type": final.get("agent_type", "unknown"),
            "time_offset_minutes": float(final.get("time_offset_minutes", 0.0)),
            "context": final.get("context", {}),
            "trace": final.get("trace", []),
        }

    def get_session_summary(self, session_id: str) -> Dict:
        obs = self.session_memory.get(session_id, {}).get("observations", [])
        summary: Dict[str, int] = {"PRESENT": 0, "LATE": 0, "ABSENT": 0, "FLAGGED": 0}
        for o in obs:
            d = o.get("decision", "FLAGGED")
            summary[d] = summary.get(d, 0) + 1
        return {
            "session_id": session_id,
            "total_observations": len(obs),
            "summary": summary,
            "observations": obs,
        }

    def batch_process_recognitions(
        self,
        recognitions: List[Dict],
        class_start_time: datetime,
        student_database: Optional[Dict] = None,
        lecture_context: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        decisions = []
        student_database = student_database or {}
        for rec in recognitions:
            d = self.make_decision(
                student_name=rec["student_name"],
                confidence_score=rec["confidence"],
                current_time=rec["detection_time"],
                class_start_time=class_start_time,
                student_history=student_database.get(rec["student_name"]),
                attendance_records=rec.get("attendance_records", []),
                lecture_context=lecture_context,
                session_id=session_id,
            )
            d["student_name"] = rec["student_name"]
            decisions.append(d)
        return decisions

    # ── private helpers ──────────────────────────────────────────────────────

    def _get_unknown_frequency(
        self, session_id: Optional[str], sig: Optional[str]
    ) -> int:
        if not session_id or not sig:
            return 0
        session = self.session_memory.setdefault(
            session_id, {"observations": [], "unknown_faces": {}}
        )
        unknown_faces = session.setdefault("unknown_faces", {})
        unknown_faces[sig] = unknown_faces.get(sig, 0) + 1
        return unknown_faces[sig]

    def _record_session(self, session_id: str, state: dict) -> None:
        self.session_memory.setdefault(
            session_id, {"observations": [], "unknown_faces": {}}
        )
        self.session_memory[session_id]["observations"].append(
            {
                "student_name": state.get("student_name"),
                "decision": state.get("decision"),
                "confidence": state.get("confidence_score"),
                "uncertainty_score": state.get("uncertainty_score"),
                "action": state.get("action"),
                "requires_review": state.get("requires_review", False),
                "agent_type": state.get("agent_type"),
                "time_offset_minutes": state.get("time_offset_minutes"),
            }
        )

    def _summarize_recent_attendance(self, records: List[Dict]) -> str:
        if not records:
            return "No historical data"
        recent = records[-5:]
        parts = []
        for r in recent:
            status = str(r.get("status", "unknown")).upper()
            at = r.get("marked_at") or r.get("time")
            parts.append(f"{status} at {at}" if at else status)
        return "Recent: " + ", ".join(parts)

    def _parse_json(self, text: str) -> Dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {}

    def _fallback_decision(self, state: dict) -> Dict:
        conf = float(state.get("confidence_score", 0))
        return {
            "decision": "FLAGGED" if conf < self.LOW_CONF else "PRESENT",
            "confidence": conf,
            "uncertainty_score": 0.5,
            "action": (
                "ESCALATE_TO_INSTRUCTOR" if conf < self.LOW_CONF else "MARK_PRESENT"
            ),
            "reasoning": "LangGraph unavailable — basic fallback used.",
            "requires_review": conf < self.LOW_CONF,
            "override_option": True,
            "agent_type": "fallback",
            "time_offset_minutes": 0.0,
            "context": {},
            "trace": ["fallback → LangGraph not installed"],
        }
