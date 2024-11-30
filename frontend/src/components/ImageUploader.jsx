import React, { useState } from 'react';

const CLOUDINARY_URL = 'https://api.cloudinary.com/v1_1/your_cloud_name/image/upload';
const UPLOAD_PRESET = 'your_upload_preset';
const BACKEND_URL = 'http://127.0.0.1:5000/generate-caption';

const ImageUploader = () => {
  const [imageUrl, setImageUrl] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('upload_preset', UPLOAD_PRESET);

    setLoading(true);

    try {
      // Upload to Cloudinary
      const cloudinaryResponse = await fetch(CLOUDINARY_URL, {
        method: 'POST',
        body: formData,
      });
      const cloudinaryData = await cloudinaryResponse.json();
      const uploadedImageUrl = cloudinaryData.secure_url;

      setImageUrl(uploadedImageUrl);

      // Send URL to backend for caption generation
      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: uploadedImageUrl }),
      });

      const data = await response.json();
      setCaption(data.caption || 'Failed to generate caption.');
    } catch (error) {
      console.error('Error:', error);
      setCaption('Error occurred while generating caption.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleFileUpload} />
      {loading && <p>Uploading and processing...</p>}
      {imageUrl && (
        <>
          <h2>Uploaded Image:</h2>
          <img src={imageUrl} alt="Uploaded" style={{ width: '300px' }} />
        </>
      )}
      {caption && (
        <>
          <h2>Generated Caption:</h2>
          <p>{caption}</p>
        </>
      )}
    </div>
  );
};

export default ImageUploader;
