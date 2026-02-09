/**
 * Frontend File Upload Example
 * 
 * Demonstrates how to upload a file to S3 using the backend's presigned URL
 * and then save metadata to Supabase.
 */

import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
const supabase = createClient(supabaseUrl, supabaseKey)

// Configuration
const API_BASE_URL = "http://localhost:8000"; // Your FastAPI URL

/**
 * Uploads a file to S3 via backend-generated presigned URL.
 * 
 * @param {File} file - The file object from <input type="file">
 * @returns {Promise<Object>} - Result object
 */
async function uploadFileToS3(file) {
    try {
        // 1. Get current session (JWT)
        const { data: { session }, error: authError } = await supabase.auth.getSession();

        if (authError || !session) {
            throw new Error("User not authenticated");
        }

        const token = session.access_token;
        console.log("Got access token");

        // 2. Request Presigned URL from Backend
        const response = await fetch(`${API_BASE_URL}/api/upload/url`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}` // Pass JWT for verification
            },
            body: JSON.stringify({
                filename: file.name,
                file_type: file.type
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Failed to get upload URL");
        }

        const { upload_url, s3_key, bucket } = await response.json();
        console.log("Got upload URL:", upload_url);

        // 3. Upload File directly to S3
        const uploadResponse = await fetch(upload_url, {
            method: "PUT",
            headers: {
                "Content-Type": file.type
            },
            body: file
        });

        if (!uploadResponse.ok) {
            throw new Error("Failed to upload to S3");
        }

        console.log("Upload successful!");

        // 4. Save Metadata to Supabase (user_files table)
        const { data: insertData, error: dbError } = await supabase
            .from('user_files')
            .insert({
                user_id: session.user.id,
                file_name: file.name,
                s3_bucket: bucket,
                s3_key: s3_key,
                file_size_bytes: file.size,
                content_type: file.type,
                // file_url: upload_url.split('?')[0] // Optional: public URL logic
            })
            .select();

        if (dbError) {
            console.error("Database error:", dbError);
            throw new Error("File uploaded but failed to save metadata");
        }

        return {
            success: true,
            message: "File uploaded successfully!",
            data: insertData[0]
        };

    } catch (error) {
        console.error("Upload error:", error);
        return { success: false, message: error.message };
    }
}

// Example Usage:
// const fileInput = document.getElementById('fileInput');
// fileInput.addEventListener('change', async (e) => {
//   const file = e.target.files[0];
//   if (file) {
//     const result = await uploadFileToS3(file);
//     alert(result.message);
//   }
// });
