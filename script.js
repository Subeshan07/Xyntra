
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const form = document.getElementById('upload-form');
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            
            if (file) {
                const reader = new FileReader();
                
                reader.addEventListener('load', function() {
                    imagePreview.setAttribute('src', this.result);
                    previewContainer.classList.remove('hidden');
                });
                
                reader.readAsDataURL(file);
            }
        });
    }
    
    if (form) {
        form.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file');
            
            if (fileInput && fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select an image file to analyze.');
            }
        });
    }
});