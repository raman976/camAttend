import os
from supabase import create_client, Client
import numpy as np
import bcrypt
import base64
from postgrest.exceptions import APIError
from datetime import date, time
from typing import List, Dict, Tuple


class SupabaseDB:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        
        self.client: Client = create_client(self.url, self.key)
    
    def create_organization(self, name: str, code: str, contact_email: str = None) -> Dict:
        data = {"name": name, "code": code, "contact_email": contact_email}
        result = self.client.table("organizations").insert(data).execute()
        return result.data[0] if result.data else None
    
    def get_organization(self, org_id: str) -> Dict:
        result = self.client.table("organizations").select("*").eq("id", org_id).execute()
        return result.data[0] if result.data else None
    
    def get_organization_by_code(self, code: str) -> Dict:
        result = self.client.table("organizations").select("*").eq("code", code).execute()
        return result.data[0] if result.data else None
    
    def enroll_student(self, organization_id: str, student_id: str, name: str, 
                      embedding: np.ndarray, email: str = None, phone: str = None,
                      department: str = None, enrollment_year: int = None,
                      photo_url: str = None) -> Dict:
        embedding_bytes = embedding.astype(np.float32).tobytes()
        embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
        
        data = {
            "organization_id": organization_id,
            "student_id": student_id,
            "name": name,
            "email": email,
            "phone": phone,
            "department": department,
            "enrollment_year": enrollment_year,
            "face_embedding": embedding_b64,
            "photo_url": photo_url
        }
        
        result = self.client.table("students").insert(data).execute()
        return result.data[0] if result.data else None
    
    def get_student(self, student_uuid: str) -> Dict:
        result = self.client.table("students").select("*").eq("id", student_uuid).execute()
        return result.data[0] if result.data else None
    
    def get_students_by_organization(self, org_id: str, active_only: bool = True) -> List[Dict]:
        query = self.client.table("students").select("*").eq("organization_id", org_id)
        if active_only:
            query = query.eq("is_active", True)
        result = query.execute()
        return result.data if result.data else []
    
    def get_student_embeddings(self, org_id: str) -> List[Tuple[str, str, np.ndarray]]:
        students = self.get_students_by_organization(org_id, active_only=True)
        embeddings = []
        
        for student in students:
            embedding_data = student['face_embedding']
            
            try:
                if isinstance(embedding_data, bytes):
                    embedding_bytes = embedding_data
                elif isinstance(embedding_data, str):
                    if embedding_data.startswith('\\x'):
                        hex_str = embedding_data[2:]
                        base64_bytes = bytes.fromhex(hex_str)
                        base64_str = base64_bytes.decode('utf-8')
                        embedding_bytes = base64.b64decode(base64_str)
                    else:
                        embedding_bytes = base64.b64decode(embedding_data)
                else:
                    continue
                
                emb = np.frombuffer(embedding_bytes, dtype=np.float32)
                if emb.shape[0] != 512:
                    continue
                
                embeddings.append((student['id'], student['name'], emb))
            except Exception:
                continue
        
        return embeddings
    
    def update_student(self, student_uuid: str, **kwargs) -> Dict:
        data = {k: v for k, v in kwargs.items() if v is not None}
        result = self.client.table("students").update(data).eq("id", student_uuid).execute()
        return result.data[0] if result.data else None
    
    def deactivate_student(self, student_uuid: str) -> Dict:
        return self.update_student(student_uuid, is_active=False)
    
    def create_lecture(self, organization_id: str, created_by: str, title: str,
                      lecture_date: date, subject: str = None, course_code: str = None,
                      start_time: time = None, end_time: time = None, 
                      location: str = None, description: str = None) -> Dict:
        data = {
            "organization_id": organization_id,
            "created_by": created_by,
            "title": title,
            "subject": subject,
            "course_code": course_code,
            "lecture_date": lecture_date.isoformat(),
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "location": location,
            "description": description,
            "status": "scheduled"
        }
        
        result = self.client.table("lectures").insert(data).execute()
        return result.data[0] if result.data else None
    
    def get_lecture(self, lecture_id: str) -> Dict:
        result = self.client.table("lectures").select("*").eq("id", lecture_id).execute()
        return result.data[0] if result.data else None
    
    def get_lectures_by_organization(self, org_id: str, status: str = None,
                                    start_date: date = None, end_date: date = None) -> List[Dict]:
        query = self.client.table("lectures").select("*").eq("organization_id", org_id)
        
        if status:
            query = query.eq("status", status)
        if start_date:
            query = query.gte("lecture_date", start_date.isoformat())
        if end_date:
            query = query.lte("lecture_date", end_date.isoformat())
        
        result = query.order("lecture_date", desc=True).execute()
        return result.data if result.data else []
    
    def update_lecture_status(self, lecture_id: str, status: str) -> Dict:
        result = self.client.table("lectures").update({"status": status}).eq("id", lecture_id).execute()
        return result.data[0] if result.data else None
    
    def mark_attendance(self, lecture_id: str, student_id: str, marked_by: str,
                       confidence_score: float = None, status: str = "present",
                       notes: str = None) -> Dict:
        data = {
            "lecture_id": lecture_id,
            "student_id": student_id,
            "marked_by": marked_by,
            "confidence_score": confidence_score,
            "status": status,
            "notes": notes
        }
        
        result = self.client.table("attendance").upsert(data).execute()
        return result.data[0] if result.data else None
    
    def mark_bulk_attendance(self, lecture_id: str, marked_by: str,
                           student_data: List[Dict]) -> List[Dict]:
        records = []
        for data in student_data:
            records.append({
                "lecture_id": lecture_id,
                "student_id": data['student_id'],
                "marked_by": marked_by,
                "confidence_score": data.get('confidence_score'),
                "status": data.get('status', 'present'),
                "notes": data.get('notes')
            })
        
        result = self.client.table("attendance").upsert(records).execute()
        return result.data if result.data else []
    
    def get_lecture_attendance(self, lecture_id: str) -> List[Dict]:
        result = self.client.table("attendance").select("*, students(*)").eq("lecture_id", lecture_id).execute()
        return result.data if result.data else []
    
    def get_attendance_report(self, lecture_id: str) -> List[Dict]:
        result = self.client.rpc("get_lecture_attendance_report", {"lecture_uuid": lecture_id}).execute()
        return result.data if result.data else []
    
    def get_attendance_stats(self, lecture_id: str) -> Dict:
        result = self.client.rpc("get_attendance_stats", {"lecture_uuid": lecture_id}).execute()
        return result.data if result.data else {}
    
    def get_student_attendance_history(self, student_id: str, start_date: date = None,
                                      end_date: date = None) -> List[Dict]:
        query = self.client.table("attendance").select("*, lectures(*)").eq("student_id", student_id)
        
        if start_date:
            query = query.gte("lectures.lecture_date", start_date.isoformat())
        if end_date:
            query = query.lte("lectures.lecture_date", end_date.isoformat())
        
        result = query.order("marked_at", desc=True).execute()
        return result.data if result.data else []
    
    def save_attendance_image(self, lecture_id: str, image_url: str, faces_detected: int,
                             students_recognized: int, uploaded_by: str) -> Dict:
        data = {
            "lecture_id": lecture_id,
            "image_url": image_url,
            "faces_detected": faces_detected,
            "students_recognized": students_recognized,
            "uploaded_by": uploaded_by
        }
        
        result = self.client.table("attendance_images").insert(data).execute()
        return result.data[0] if result.data else None
    
    def get_lecture_images(self, lecture_id: str) -> List[Dict]:
        result = self.client.table("attendance_images").select("*").eq(
            "lecture_id", lecture_id).order("uploaded_at", desc=True).execute()
        return result.data if result.data else []
    
    def create_user(self, organization_id: str, email: str, name: str,
                   password: str, role: str = "instructor") -> Dict:
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=4)).decode('utf-8')
        
        data = {
            "organization_id": organization_id,
            "email": email,
            "name": name,
            "password_hash": password_hash,
            "role": role
        }
        
        try:
            result = self.client.table("users").insert(data).execute()
            return result.data[0] if result.data else None
        except APIError as e:
            # Duplicate email is a common registration case; return None for graceful UI handling.
            if str(getattr(e, "message", "")).find("users_email_key") != -1 or str(e).find("users_email_key") != -1:
                return None
            raise
    
    def get_user_by_email(self, email: str) -> Dict:
        result = self.client.table("users").select("*").eq("email", email).execute()
        return result.data[0] if result.data else None
    
    def verify_user_password(self, email: str, password: str) -> Dict:
        user = self.get_user_by_email(email)
        
        if not user or not user.get('password_hash'):
            return None
        
        if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return user
        
        return None
    
    def export_attendance_csv(self, lecture_id: str) -> str:
        report = self.get_attendance_report(lecture_id)
        
        if not report:
            return ""
        
        csv_lines = ["Student ID,Name,Email,Status,Marked At,Confidence Score"]
        
        for record in report:
            csv_lines.append(
                f"{record['student_id']},{record['student_name']},"
                f"{record['email'] or ''},"
                f"{record['status']},"
                f"{record['marked_at'] or ''},"
                f"{record['confidence_score'] or ''}"
            )
        
        return "\n".join(csv_lines)
    
    # Agent-specific methods
    
    def save_agent_decision(self, lecture_id: str, student_id: str, student_name: str,
                           face_confidence: float, agent_decision: str, agent_reasoning: str,
                           agent_type: str, time_offset_minutes: float = None,
                           requires_review: bool = False, admin_override: str = None) -> Dict:
        """Save agent's attendance decision"""
        data = {
            "lecture_id": lecture_id,
            "student_id": student_id,
            "student_name": student_name,
            "face_confidence": face_confidence,
            "agent_decision": agent_decision,
            "agent_reasoning": agent_reasoning,
            "agent_type": agent_type,
            "time_offset_minutes": time_offset_minutes,
            "requires_review": requires_review,
            "admin_override": admin_override,
            "status": "pending_review" if requires_review else "auto_marked"
        }
        
        result = self.client.table("agent_decisions").insert(data).execute()
        return result.data[0] if result.data else None
    
    def get_student_attendance_stats(self, student_id: str, organization_id: str,
                                    limit_days: int = 30) -> Dict:
        """Get student's attendance history and statistics for agent reasoning"""
        history = self.get_student_attendance_history(student_id)
        
        if not history:
            return {
                'total_classes': 0,
                'present': 0,
                'late': 0,
                'absent': 0,
                'avg_attendance': 0.0,
                'attendance_percentage': 0.0,
                'previous_recognition_errors': 0,
                'recent_pattern': []
            }
        
        # Count by status
        present_count = sum(1 for h in history if h.get('status') == 'present')
        late_count = sum(1 for h in history if h.get('status') == 'late')
        absent_count = sum(1 for h in history if h.get('status') == 'absent')
        total_count = len(history)
        
        avg_attendance = (present_count + late_count) / total_count if total_count > 0 else 0
        attendance_percentage = round(avg_attendance * 100.0, 2)
        
        # Get recent pattern (last 10 records)
        recent_pattern = [h.get('status', 'unknown') for h in history[:10]]
        
        # Optional: read prior recognition errors from agent decisions if table exists.
        previous_recognition_errors = 0
        try:
            errors_result = self.client.table("agent_decisions").select("id", count="exact").eq(
                "student_id", student_id
            ).eq("requires_review", True).execute()
            previous_recognition_errors = int(errors_result.count or 0)
        except Exception:
            previous_recognition_errors = 0

        # Optional: persist attendance percentage in students table if column exists.
        try:
            self.client.table("students").update({
                "attendance_percentage": attendance_percentage
            }).eq("id", student_id).execute()
        except Exception:
            pass

        return {
            'total_classes': total_count,
            'present': present_count,
            'late': late_count,
            'absent': absent_count,
            'avg_attendance': avg_attendance,
            'attendance_percentage': attendance_percentage,
            'previous_recognition_errors': previous_recognition_errors,
            'recent_pattern': recent_pattern
        }
    
    def get_flagged_decisions(self, lecture_id: str) -> List[Dict]:
        """Get all decisions flagged for human review"""
        result = self.client.table("agent_decisions").select("*").eq(
            "lecture_id", lecture_id).eq("requires_review", True).execute()
        return result.data if result.data else []
    
    def override_agent_decision(self, decision_id: str, override_status: str,
                               override_by: str, override_reason: str = None) -> Dict:
        """Override agent decision by human admin"""
        data = {
            "admin_override": override_status,
            "override_by": override_by,
            "override_reason": override_reason,
            "status": "manual_override"
        }
        
        result = self.client.table("agent_decisions").update(data).eq("id", decision_id).execute()
        return result.data[0] if result.data else None
    
    def get_agent_performance_stats(self, organization_id: str) -> Dict:
        """Analyze agent decision accuracy and patterns"""
        # This would typically be done with an RPC function
        # For now, return a simple structure
        return {
            'total_decisions': 0,
            'flagged_count': 0,
            'overridden_count': 0,
            'accuracy': 0.0
        }
