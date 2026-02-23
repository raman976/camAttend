-- Supabase Schema for Production Face Attendance System

-- Organizations table
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    code TEXT UNIQUE NOT NULL,
    contact_email TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users/Instructors table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'instructor', 'viewer')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Students table with face embeddings
CREATE TABLE students (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    student_id TEXT NOT NULL,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    enrollment_year INTEGER,
    department TEXT,
    face_embedding BYTEA NOT NULL,
    photo_url TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(organization_id, student_id)
);

-- Lectures/Classes table
CREATE TABLE lectures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_by UUID REFERENCES users(id),
    title TEXT NOT NULL,
    subject TEXT,
    course_code TEXT,
    lecture_date DATE NOT NULL,
    start_time TIME,
    end_time TIME,
    location TEXT,
    description TEXT,
    status TEXT DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'ongoing', 'completed', 'cancelled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Attendance records table
CREATE TABLE attendance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lecture_id UUID REFERENCES lectures(id) ON DELETE CASCADE,
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    marked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confidence_score FLOAT,
    status TEXT DEFAULT 'present' CHECK (status IN ('present', 'late', 'absent')),
    marked_by UUID REFERENCES users(id),
    notes TEXT,
    UNIQUE(lecture_id, student_id)
);

-- Attendance images (optional - store group photos)
CREATE TABLE attendance_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lecture_id UUID REFERENCES lectures(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    faces_detected INTEGER,
    students_recognized INTEGER,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    uploaded_by UUID REFERENCES users(id)
);

-- Indexes for better query performance
CREATE INDEX idx_students_org ON students(organization_id);
CREATE INDEX idx_students_active ON students(is_active);
CREATE INDEX idx_lectures_org ON lectures(organization_id);
CREATE INDEX idx_lectures_date ON lectures(lecture_date);
CREATE INDEX idx_lectures_status ON lectures(status);
CREATE INDEX idx_attendance_lecture ON attendance(lecture_id);
CREATE INDEX idx_attendance_student ON attendance(student_id);
CREATE INDEX idx_users_org ON users(organization_id);

-- Enable Row Level Security (RLS)
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE students ENABLE ROW LEVEL SECURITY;
ALTER TABLE lectures ENABLE ROW LEVEL SECURITY;
ALTER TABLE attendance ENABLE ROW LEVEL SECURITY;
ALTER TABLE attendance_images ENABLE ROW LEVEL SECURITY;

-- RLS Policies (basic examples - customize based on your auth setup)

-- Organizations: Users can only see their own organization
CREATE POLICY "Users can view own organization"
    ON organizations FOR SELECT
    USING (id IN (SELECT organization_id FROM users WHERE id = auth.uid()));

-- Students: Users can only access students from their organization
CREATE POLICY "Users can view own org students"
    ON students FOR SELECT
    USING (organization_id IN (SELECT organization_id FROM users WHERE id = auth.uid()));

CREATE POLICY "Admins and instructors can insert students"
    ON students FOR INSERT
    WITH CHECK (
        organization_id IN (
            SELECT organization_id FROM users 
            WHERE id = auth.uid() AND role IN ('admin', 'instructor')
        )
    );

-- Lectures: Users can only access lectures from their organization
CREATE POLICY "Users can view own org lectures"
    ON lectures FOR SELECT
    USING (organization_id IN (SELECT organization_id FROM users WHERE id = auth.uid()));

CREATE POLICY "Instructors can create lectures"
    ON lectures FOR INSERT
    WITH CHECK (
        organization_id IN (
            SELECT organization_id FROM users 
            WHERE id = auth.uid() AND role IN ('admin', 'instructor')
        )
    );

-- Attendance: Users can view and mark attendance for their org
CREATE POLICY "Users can view own org attendance"
    ON attendance FOR SELECT
    USING (
        lecture_id IN (
            SELECT id FROM lectures 
            WHERE organization_id IN (
                SELECT organization_id FROM users WHERE id = auth.uid()
            )
        )
    );

CREATE POLICY "Instructors can mark attendance"
    ON attendance FOR INSERT
    WITH CHECK (
        lecture_id IN (
            SELECT id FROM lectures 
            WHERE organization_id IN (
                SELECT organization_id FROM users 
                WHERE id = auth.uid() AND role IN ('admin', 'instructor')
            )
        )
    );

-- Functions

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_students_updated_at BEFORE UPDATE ON students
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_lectures_updated_at BEFORE UPDATE ON lectures
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to get attendance report
CREATE OR REPLACE FUNCTION get_lecture_attendance_report(lecture_uuid UUID)
RETURNS TABLE (
    student_name TEXT,
    student_id TEXT,
    email TEXT,
    status TEXT,
    marked_at TIMESTAMP WITH TIME ZONE,
    confidence_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.name,
        s.student_id,
        s.email,
        COALESCE(a.status, 'absent')::TEXT,
        a.marked_at,
        a.confidence_score
    FROM students s
    LEFT JOIN attendance a ON s.id = a.student_id AND a.lecture_id = lecture_uuid
    WHERE s.organization_id = (SELECT organization_id FROM lectures WHERE id = lecture_uuid)
    AND s.is_active = true
    ORDER BY s.name;
END;
$$ LANGUAGE plpgsql;

-- Function to get attendance statistics
CREATE OR REPLACE FUNCTION get_attendance_stats(lecture_uuid UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_students', COUNT(DISTINCT s.id),
        'present', COUNT(DISTINCT CASE WHEN a.status = 'present' THEN a.student_id END),
        'late', COUNT(DISTINCT CASE WHEN a.status = 'late' THEN a.student_id END),
        'absent', COUNT(DISTINCT s.id) - COUNT(DISTINCT a.student_id),
        'attendance_percentage', 
            ROUND(
                (COUNT(DISTINCT a.student_id)::DECIMAL / NULLIF(COUNT(DISTINCT s.id), 0) * 100), 2
            )
    ) INTO result
    FROM students s
    LEFT JOIN attendance a ON s.id = a.student_id AND a.lecture_id = lecture_uuid
    WHERE s.organization_id = (SELECT organization_id FROM lectures WHERE id = lecture_uuid)
    AND s.is_active = true;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;
