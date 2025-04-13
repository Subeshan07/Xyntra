document.addEventListener('DOMContentLoaded', function() {
    // Original file input preview code
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
                    
                    // Add animation class to preview
                    imagePreview.classList.add('fade-in');
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
            } else {
                // Show loading state if form is valid
                const submitBtn = this.querySelector('.submit-btn');
                if (submitBtn) {
                    submitBtn.innerHTML = '<div class="loading-spinner"></div> Analyzing...';
                    submitBtn.disabled = true;
                }
            }
        });
    }
    
    // Add tooltips for truncated text
    const barLabels = document.querySelectorAll('.bar-label');
    if (barLabels) {
        barLabels.forEach(label => {
            if (label.offsetWidth < label.scrollWidth) {
                label.title = label.textContent;
            }
        });
    }
    
    // Animate bars on results page
    const bars = document.querySelectorAll('.bar');
    if (bars.length > 0) {
        // Animate bars sequentially
        bars.forEach((bar, index) => {
            setTimeout(() => {
                bar.style.width = bar.style.width;
                bar.classList.add('animated');
            }, 200 * index);
        });
    }
});

// Doctor Chat and Scheduling JavaScript
// Add this to the bottom of script.js or as a new file

document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    if (tabBtns.length > 0) {
        tabBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                // Remove active class from all buttons and panes
                tabBtns.forEach(b => b.classList.remove('active'));
                tabPanes.forEach(p => p.classList.remove('active'));
                
                // Add active class to clicked button and corresponding pane
                btn.classList.add('active');
                const tabId = btn.getAttribute('data-tab');
                document.getElementById(`${tabId}-pane`).classList.add('active');
            });
        });
    }
    
    // Chat functionality
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-message');
    
    if (chatMessages && messageInput && sendButton) {
        // Send message function
        function sendMessage() {
            const message = messageInput.value.trim();
            if (message) {
                // Add user message
                addMessage(message, 'user');
                
                // Clear input
                messageInput.value = '';
                
                // Simulate doctor response after delay
                setTimeout(() => {
                    let response;
                    const cancerType = document.querySelector('.cancer-type-name').textContent;
                    
                    // Customize response based on detected cancer type
                    if (message.toLowerCase().includes('treatment') || message.toLowerCase().includes('options')) {
                        if (cancerType.includes('large.cell.carcinoma')) {
                            response = "Based on the large cell carcinoma findings, your treatment options may include surgery, radiation therapy, chemotherapy, or a combination. The exact approach depends on the stage of cancer and your overall health. I'd recommend scheduling an appointment so we can discuss the specifics in detail.";
                        } else if (cancerType.includes('squamous.cell.carcinoma')) {
                            response = "For squamous cell carcinoma, we typically consider surgery for early stages, possibly followed by radiation. Advanced cases might require chemotherapy and immunotherapy. Let's schedule a consultation to develop a personalized treatment plan.";
                        } else {
                            response = "Treatment options for your type of lung cancer typically include surgery, radiation therapy, chemotherapy, targeted therapy, and immunotherapy. The best approach depends on several factors including stage, location, and your overall health. I'd like to discuss this in detail during an appointment.";
                        }
                    } 
                    else if (message.toLowerCase().includes('prognosis') || message.toLowerCase().includes('survival')) {
                        response = "Prognosis depends on many factors including the stage, your overall health, and how well the cancer responds to treatment. It's best to discuss this in person where we can review all your specific details. Would you like to schedule an appointment?";
                    }
                    else if (message.toLowerCase().includes('appointment') || message.toLowerCase().includes('schedule') || message.toLowerCase().includes('meet')) {
                        response = "I'd be happy to meet with you. You can schedule an appointment using the scheduling tab. Just select a date and time that works for you, and I'll make sure we have enough time to address all your questions.";
                        
                        // Switch to scheduling tab
                        setTimeout(() => {
                            document.querySelector('.tab-btn[data-tab="schedule"]').click();
                        }, 1000);
                    }
                    else {
                        response = "Thank you for your message. I understand this can be concerning. The scan analysis provides important information, but a complete evaluation requires a detailed discussion of your medical history and possibly additional tests. Would you like to schedule an appointment to discuss next steps?";
                    }
                    
                    addMessage(response, 'doctor');
                    
                    // Scroll to bottom
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }, 1000);
                
                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
        
        // Add message to chat
        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', sender);
            
            const avatar = document.createElement('div');
            avatar.classList.add('message-avatar');
            avatar.textContent = sender === 'user' ? 'You' : 'Dr';
            
            const content = document.createElement('div');
            content.classList.add('message-content');
            
            const messageText = document.createElement('p');
            messageText.textContent = text;
            
            const time = document.createElement('span');
            time.classList.add('message-time');
            time.textContent = 'Just now';
            
            content.appendChild(messageText);
            content.appendChild(time);
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            
            chatMessages.appendChild(messageDiv);
        }
        
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Scheduling functionality
    const doctorCards = document.querySelectorAll('.doctor-card');
    const calendarGrid = document.getElementById('calendar-grid');
    const currentMonthDisplay = document.getElementById('current-month');
    const prevMonthBtn = document.getElementById('prev-month');
    const nextMonthBtn = document.getElementById('next-month');
    const timeSlotContainer = document.getElementById('time-slots');
    const appointmentSummary = document.getElementById('appointment-summary');
    const appointmentConfirmed = document.getElementById('appointment-confirmed');
    const confirmAppointmentBtn = document.getElementById('confirm-appointment');
    
    // Track selected items
    let selectedDoctor = null;
    let selectedDate = null;
    let selectedTime = null;
    let currentDate = new Date();
    
    // Initialize scheduling components
    if (doctorCards.length > 0 && calendarGrid) {
        // Doctor selection
        doctorCards.forEach(card => {
            card.addEventListener('click', function() {
                doctorCards.forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                selectedDoctor = {
                    id: card.getAttribute('data-doctor-id'),
                    name: card.querySelector('h4').textContent
                };
                updateAppointmentSummary();
            });
        });
        
        // Calendar navigation
        if (prevMonthBtn && nextMonthBtn) {
            prevMonthBtn.addEventListener('click', function() {
                currentDate.setMonth(currentDate.getMonth() - 1);
                renderCalendar();
            });
            
            nextMonthBtn.addEventListener('click', function() {
                currentDate.setMonth(currentDate.getMonth() + 1);
                renderCalendar();
            });
        }
        
        // Render initial calendar
        renderCalendar();
        
        // Handle confirm appointment
        if (confirmAppointmentBtn) {
            confirmAppointmentBtn.addEventListener('click', function() {
                if (selectedDoctor && selectedDate && selectedTime) {
                    // Display confirmation
                    document.getElementById('confirmed-doctor').textContent = selectedDoctor.name;
                    document.getElementById('confirmed-date-time').textContent = 
                        `${formatDate(selectedDate)} at ${selectedTime}`;
                    
                    appointmentSummary.classList.add('hidden');
                    appointmentConfirmed.classList.remove('hidden');
                    
                    // Scroll to confirmation
                    appointmentConfirmed.scrollIntoView({ behavior: 'smooth' });
                }
            });
        }
    }
    
    // Calendar rendering
    function renderCalendar() {
        // Clear previous calendar
        calendarGrid.innerHTML = '';
        
        // Update month display
        const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December'];
        currentMonthDisplay.textContent = `${monthNames[currentDate.getMonth()]} ${currentDate.getFullYear()}`;
        
        // Add day headers
        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        dayNames.forEach(day => {
            const dayHeader = document.createElement('div');
            dayHeader.classList.add('calendar-day', 'header');
            dayHeader.textContent = day;
            calendarGrid.appendChild(dayHeader);
        });
        
        // Get first day of month and number of days
        const firstDay = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
        const lastDay = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0);
        
        // Add empty cells for days before first day of month
        for (let i = 0; i < firstDay.getDay(); i++) {
            const emptyDay = document.createElement('div');
            emptyDay.classList.add('calendar-day', 'other-month');
            calendarGrid.appendChild(emptyDay);
        }
        
        // Add days of current month
        const today = new Date();
        for (let i = 1; i <= lastDay.getDate(); i++) {
            const dayCell = document.createElement('div');
            dayCell.classList.add('calendar-day');
            dayCell.textContent = i;
            
            const currentDateObj = new Date(currentDate.getFullYear(), currentDate.getMonth(), i);
            
            // Check if date is in the past
            if (currentDateObj < today && 
                !(currentDateObj.getDate() === today.getDate() && 
                  currentDateObj.getMonth() === today.getMonth() && 
                  currentDateObj.getFullYear() === today.getFullYear())) {
                dayCell.classList.add('unavailable');
            } else {
                dayCell.classList.add('selectable');
                
                // Weekend availability (simulate some unavailable days)
                if (currentDateObj.getDay() === 0) { // Sunday
                    dayCell.classList.add('unavailable');
                    dayCell.classList.remove('selectable');
                } else {
                    // Add click event
                    dayCell.addEventListener('click', function() {
                        if (!dayCell.classList.contains('unavailable')) {
                            // Remove selected class from all days
                            document.querySelectorAll('.calendar-day.selected').forEach(day => {
                                day.classList.remove('selected');
                            });
                            
                            // Add selected class to clicked day
                            dayCell.classList.add('selected');
                            
                            // Set selected date
                            selectedDate = new Date(currentDate.getFullYear(), currentDate.getMonth(), i);
                            
                            // Update time slots
                            renderTimeSlots();
                            
                            // Update appointment summary
                            updateAppointmentSummary();
                        }
                    });
                }
            }
            
            // Mark selected date
            if (selectedDate && 
                i === selectedDate.getDate() && 
                currentDate.getMonth() === selectedDate.getMonth() && 
                currentDate.getFullYear() === selectedDate.getFullYear()) {
                dayCell.classList.add('selected');
            }
            
            calendarGrid.appendChild(dayCell);
        }
    }
    
    // Render time slots based on selected date
    function renderTimeSlots() {
        if (!timeSlotContainer) return;
        
        // Clear previous time slots
        timeSlotContainer.innerHTML = '';
        
        // Generate available time slots
        // In a real app, these would come from a database based on doctor availability
        const timeSlots = ['9:00 AM', '9:30 AM', '10:00 AM', '10:30 AM', 
                          '11:00 AM', '11:30 AM', '1:00 PM', '1:30 PM', 
                          '2:00 PM', '2:30 PM', '3:00 PM', '3:30 PM'];
        
        // Generate some random unavailable slots
        const unavailableSlots = [];
        for (let i = 0; i < 3; i++) {
            const randomIndex = Math.floor(Math.random() * timeSlots.length);
            unavailableSlots.push(timeSlots[randomIndex]);
        }
        
        timeSlots.forEach(time => {
            const timeSlot = document.createElement('div');
            timeSlot.classList.add('time-slot');
            timeSlot.textContent = time;
            
            // Check if slot is unavailable
            if (unavailableSlots.includes(time)) {
                timeSlot.classList.add('unavailable');
            } else {
                // Add click event
                timeSlot.addEventListener('click', function() {
                    if (!timeSlot.classList.contains('unavailable')) {
                        // Remove selected class from all time slots
                        document.querySelectorAll('.time-slot.selected').forEach(slot => {
                            slot.classList.remove('selected');
                        });
                        
                        // Add selected class to clicked time slot
                        timeSlot.classList.add('selected');
                        
                        // Set selected time
                        selectedTime = time;
                        
                        // Update appointment summary
                        updateAppointmentSummary();
                    }
                });
            }
            
            // Mark selected time
            if (selectedTime === time) {
                timeSlot.classList.add('selected');
            }
            
            timeSlotContainer.appendChild(timeSlot);
        });
    }
    
    // Update appointment summary
    function updateAppointmentSummary() {
        if (!appointmentSummary) return;
        
        const summaryDoctor = document.getElementById('summary-doctor');
        const summaryDate = document.getElementById('summary-date');
        const summaryTime = document.getElementById('summary-time');
        
        if (summaryDoctor && selectedDoctor) {
            summaryDoctor.textContent = selectedDoctor.name;
        }
        
        if (summaryDate && selectedDate) {
            summaryDate.textContent = formatDate(selectedDate);
        }
        
        if (summaryTime && selectedTime) {
            summaryTime.textContent = selectedTime;
        }
        
        // Show or hide summary based on selections
        if (selectedDoctor && selectedDate && selectedTime) {
            appointmentSummary.classList.remove('hidden');
        } else {
            appointmentSummary.classList.add('hidden');
        }
    }
    
    // Helper function to format date
    function formatDate(date) {
        if (!date) return '';
        
        const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December'];
        
        const day = date.getDate();
        const month = monthNames[date.getMonth()];
        const year = date.getFullYear();
        
        return `${month} ${day}, ${year}`;
    }
    
    // Initialize the current date and render calendar
    renderCalendar();
    
    // Auto-select first doctor to improve UX
    if (doctorCards.length > 0) {
        doctorCards[0].click();
    }
    
    // For demo purposes, let's pre-select a date that is available
    // This would make the flow smoother for first-time users
    setTimeout(() => {
        // Find a valid date (not in past, not unavailable)
        const availableDays = document.querySelectorAll('.calendar-day.selectable:not(.unavailable)');
        if (availableDays.length > 0) {
            // Click the first available day
            availableDays[0].click();
        }
    }, 500);
});