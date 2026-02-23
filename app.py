import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from dotenv import load_dotenv

from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.matcher import FaceMatcher
from database.supabase_db import SupabaseDB

load_dotenv()


class LoginWindow:
    def __init__(self, root, db):
        self.root = root
        self.db = db
        self.admin_user = None
        self.organization = None
        
        self.root.title("Admin Login - Face Attendance System")
        self.root.geometry("400x300")
        
        self.create_ui()
    
    def create_ui(self):
        # Center frame
        center_frame = ttk.Frame(self.root, padding="20")
        center_frame.pack(expand=True)
        
        ttk.Label(center_frame, text="Face Attendance System", font=("Arial", 16, "bold")).pack(pady=20)
        
        # Full Name (for registration)
        ttk.Label(center_frame, text="Full Name (for registration):").pack(anchor='w')
        self.name_entry = ttk.Entry(center_frame, width=30)
        self.name_entry.pack(pady=(0, 10))
        
        ttk.Label(center_frame, text="Email:").pack(anchor='w')
        self.email_entry = ttk.Entry(center_frame, width=30)
        self.email_entry.pack(pady=(0, 10))
        
        ttk.Label(center_frame, text="Password:").pack(anchor='w')
        self.password_entry = ttk.Entry(center_frame, width=30, show="*")
        self.password_entry.pack(pady=(0, 10))
        
        ttk.Label(center_frame, text="Organization Code:").pack(anchor='w')
        self.org_code_entry = ttk.Entry(center_frame, width=30)
        self.org_code_entry.pack(pady=(0, 5))
        
        # Hint text
        ttk.Label(center_frame, text="Use existing code: TEST_0 or mit", font=("Arial", 9), foreground="gray").pack(pady=(0, 15))
        
        btn_frame = ttk.Frame(center_frame)
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="Login", command=self.login).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Register", command=self.register).pack(side='left', padx=5)
        
        self.status_label = ttk.Label(center_frame, text="", foreground="red")
        self.status_label.pack(pady=10)
    
    def login(self):
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        org_code = self.org_code_entry.get().strip()
        
        if not email or not password or not org_code:
            self.status_label.config(text="Please fill Email, Password, and Org Code", foreground="red")
            return
        
        try:
            # Get organization
            org = self.db.get_organization_by_code(org_code)
            if not org:
                self.status_label.config(text="Organization not found", foreground="red")
                return
            
            # Verify user credentials
            user = self.db.verify_user_password(email, password)
            if not user:
                self.status_label.config(text="Invalid email or password", foreground="red")
                return
            
            # Check if user belongs to this organization
            if user['organization_id'] != org['id']:
                self.status_label.config(text="You don't belong to this organization", foreground="red")
                return
            
            # Success
            self.admin_user = user
            self.organization = org
            messagebox.showinfo("Success", f"Welcome {user['name']}!\nOrganization: {org['name']}")
            self.root.destroy()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
    
    def register(self):
        name = self.name_entry.get().strip()
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        org_code = self.org_code_entry.get().strip()
        
        # Validate all fields
        if not name or not email or not password or not org_code:
            self.status_label.config(text="Please fill all fields for registration", foreground="red")
            return
        
        # Validate password strength
        if len(password) < 6:
            self.status_label.config(text="Password must be at least 6 characters", foreground="red")
            return
        
        try:
            self.status_label.config(text="Processing registration...", foreground="blue")
            self.root.update()
            
            # Get or create organization
            org = self.db.get_organization_by_code(org_code)
            if not org:
                # Ask for organization name
                org_name = simpledialog.askstring(
                    "New Organization",
                    f"Organization '{org_code}' doesn't exist.\nEnter organization name to create it:",
                    parent=self.root
                )
                if not org_name:
                    self.status_label.config(text="Registration cancelled", foreground="red")
                    return
                
                org = self.db.create_organization(
                    name=org_name,
                    code=org_code,
                    contact_email=email
                )
            
            # Create user with password
            self.status_label.config(text="Creating account...", foreground="blue")
            self.root.update()
            
            user = self.db.create_user(
                organization_id=org['id'],
                email=email,
                name=name,
                password=password,
                role='admin'
            )
            
            if user:
                self.status_label.config(text="Registration successful!", foreground="green")
                messagebox.showinfo("Success", f"Admin account created!\nYou can now login.")
                # Clear password field for security
                self.password_entry.delete(0, tk.END)
            else:
                self.status_label.config(text="Registration failed", foreground="red")
                
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")


class FaceAttendanceApp:
    def __init__(self, root, admin_user, organization, db):
        self.root = root
        self.admin_user = admin_user
        self.organization = organization
        self.db = db
        
        self.root.title(f"Face Attendance - {organization['name']}")
        self.root.geometry("900x600")
        
        # Initialize
        try:
            self.detector = FaceDetector()
            self.embedder = FaceEmbedder()
            self.matcher = FaceMatcher()
            
            # Load students for this organization
            self.load_organization_students()
            
            # Create UI
            self.create_ui()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize:\n{str(e)}")
            self.root.destroy()
    
    def load_organization_students(self):
        """Load all students for the organization into matcher"""
        embeddings = self.db.get_student_embeddings(self.organization['id'])
        
        for student_id, name, emb in embeddings:
            self.matcher.add_embedding(emb, name)
        
        print(f"Loaded {len(embeddings)} students from {self.organization['name']}")
    
    def create_ui(self):
        # Top bar with org info
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(
            top_frame,
            text=f"Organization: {self.organization['name']} | Admin: {self.admin_user['name']}",
            font=("Arial", 10)
        ).pack(side='left')
        
        ttk.Button(top_frame, text="Logout", command=self.logout).pack(side='right')
        
        # Notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Enrollment
        enroll_frame = ttk.Frame(notebook)
        notebook.add(enroll_frame, text='Enroll Student')
        self.create_enroll_tab(enroll_frame)
        
        # Tab 2: Recognition
        recog_frame = ttk.Frame(notebook)
        notebook.add(recog_frame, text='Recognize Faces')
        self.create_recog_tab(recog_frame)
    
    def logout(self):
        """Logout and return to login screen"""
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            self.root.destroy()
            # Restart login
            login_root = tk.Tk()
            login_window = LoginWindow(login_root, self.db)
            login_root.mainloop()
            
            if login_window.admin_user and login_window.organization:
                app_root = tk.Tk()
                app = FaceAttendanceApp(
                    app_root,
                    login_window.admin_user,
                    login_window.organization,
                    self.db
                )
                app_root.mainloop()
    
    def create_enroll_tab(self, parent):
        # Left frame
        left_frame = ttk.Frame(parent)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(left_frame, text="Student Name:").pack(anchor='w', pady=(10, 0))
        self.enroll_name = ttk.Entry(left_frame, width=40)
        self.enroll_name.pack(pady=(0, 10))
        
        ttk.Button(left_frame, text="Select Image", command=self.select_enroll_image).pack(pady=5)
        ttk.Button(left_frame, text="Enroll Student", command=self.enroll_student).pack(pady=5)
        
        self.enroll_status = tk.Text(left_frame, height=10, width=50)
        self.enroll_status.pack(pady=10)
        
        # Right frame for image preview
        right_frame = ttk.Frame(parent)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.enroll_image_label = ttk.Label(right_frame, text="No image selected")
        self.enroll_image_label.pack(expand=True)
        
        self.enroll_image = None
    
    def create_recog_tab(self, parent):
        # Left frame
        left_frame = ttk.Frame(parent)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(left_frame, text="Threshold:").pack(anchor='w')
        self.threshold = tk.DoubleVar(value=0.5)
        threshold_scale = ttk.Scale(left_frame, from_=0.3, to=0.9, variable=self.threshold, orient='horizontal')
        threshold_scale.pack(fill='x', pady=5)
        
        self.threshold_label = ttk.Label(left_frame, text=f"Threshold: {self.threshold.get():.2f}")
        self.threshold_label.pack()
        threshold_scale.configure(command=lambda v: self.threshold_label.configure(text=f"Threshold: {float(v):.2f}"))
        
        ttk.Button(left_frame, text="Select Image", command=self.select_recog_image).pack(pady=10)
        ttk.Button(left_frame, text="Recognize Faces", command=self.recognize_faces).pack(pady=5)
        
        self.recog_status = tk.Text(left_frame, height=15, width=50)
        self.recog_status.pack(pady=10)
        
        # Right frame for image
        right_frame = ttk.Frame(parent)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.recog_image_label = ttk.Label(right_frame, text="No image selected")
        self.recog_image_label.pack(expand=True)
        
        self.recog_image = None
    
    def select_enroll_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if filepath:
            self.enroll_image = cv2.imread(filepath)
            self.display_image(filepath, self.enroll_image_label)
    
    def select_recog_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if filepath:
            self.recog_image = cv2.imread(filepath)
            self.display_image(filepath, self.recog_image_label)
    
    def display_image(self, filepath, label):
        img = Image.open(filepath)
        img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo, text="")
        label.image = photo
    
    def enroll_student(self):
        if self.enroll_image is None:
            messagebox.showerror("Error", "Please select an image")
            return
        
        name = self.enroll_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a student name")
            return
        
        self.enroll_status.delete(1.0, tk.END)
        self.enroll_status.insert(tk.END, "Processing...\n")
        self.root.update()
        
        # Detect faces
        boxes, probs, landmarks = self.detector.detect_faces(self.enroll_image)
        
        if len(boxes) == 0:
            self.enroll_status.insert(tk.END, "❌ No face detected\n")
            return
        
        if len(boxes) > 1:
            self.enroll_status.insert(tk.END, f"❌ Multiple faces detected ({len(boxes)})\n")
            self.enroll_status.insert(tk.END, "Please use an image with only one face\n")
            return
        
        # Get embedding
        emb = self.embedder.get_embedding(self.enroll_image, bbox=boxes[0], landmark=landmarks[0])
        
        if emb is None:
            self.enroll_status.insert(tk.END, "❌ Failed to extract face embedding\n")
            return
        
        # Save to database (organization-scoped)
        try:
            # Generate student ID
            student_id = name.replace(" ", "_").upper()
            
            # Enroll in Supabase (only to this admin's organization)
            student = self.db.enroll_student(
                organization_id=self.organization['id'],
                student_id=student_id,
                name=name,
                embedding=emb
            )
            
            if student:
                self.matcher.add_embedding(emb, name)
                self.enroll_status.insert(tk.END, f"✓ {name} enrolled successfully!\n")
                self.enroll_status.insert(tk.END, f"Student ID: {student_id}\n")
                self.enroll_status.insert(tk.END, f"Organization: {self.organization['name']}\n")
                self.enroll_name.delete(0, tk.END)
            else:
                self.enroll_status.insert(tk.END, "❌ Failed to enroll student\n")
        except Exception as e:
            self.enroll_status.insert(tk.END, f"❌ Error: {str(e)}\n")
    
    def recognize_faces(self):
        if self.recog_image is None:
            messagebox.showerror("Error", "Please select an image")
            return
        
        self.recog_status.delete(1.0, tk.END)
        self.recog_status.insert(tk.END, "Processing...\n")
        self.root.update()
        
        # Detect faces
        boxes, probs, landmarks = self.detector.detect_faces(self.recog_image)
        
        if len(boxes) == 0:
            self.recog_status.insert(tk.END, "❌ No faces detected\n")
            return
        
        present_students = set()
        threshold = self.threshold.get()
        
        img = self.recog_image.copy()
        
        for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
            # Get embedding
            emb = self.embedder.get_embedding(self.recog_image, bbox=box, landmark=landmark)
            
            if emb is None:
                continue
            
            # Match
            student, score = self.matcher.match(emb)
            
            x1, y1, x2, y2 = map(int, box)
            
            if score > threshold:
                # Recognized
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, student, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                self.recog_status.insert(tk.END, f"Face {i+1}: {student} (score: {score:.3f})\n")
                present_students.add(student)
            else:
                # Unknown
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                self.recog_status.insert(tk.END, f"Face {i+1}: Unknown\n")
        
        # Update image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img_pil)
        self.recog_image_label.configure(image=photo, text="")
        self.recog_image_label.image = photo
        
        # Summary
        self.recog_status.insert(tk.END, f"\n--- Summary ---\n")
        self.recog_status.insert(tk.END, f"Organization: {self.organization['name']}\n")
        self.recog_status.insert(tk.END, f"Faces detected: {len(boxes)}\n")
        self.recog_status.insert(tk.END, f"Students recognized: {len(present_students)}\n\n")
        
        if present_students:
            self.recog_status.insert(tk.END, "Present:\n")
            for student in sorted(present_students):
                self.recog_status.insert(tk.END, f"  • {student}\n")


if __name__ == "__main__":
    # Initialize database
    try:
        db = SupabaseDB()
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Database Error", f"Failed to connect to Supabase:\n{str(e)}")
        exit(1)
    
    # Show login window
    login_root = tk.Tk()
    login_window = LoginWindow(login_root, db)
    login_root.mainloop()
    
    # If login successful, show main app
    if login_window.admin_user and login_window.organization:
        app_root = tk.Tk()
        app = FaceAttendanceApp(
            app_root,
            login_window.admin_user,
            login_window.organization,
            db
        )
        app_root.mainloop()


