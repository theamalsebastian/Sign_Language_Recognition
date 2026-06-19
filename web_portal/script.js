document.addEventListener('DOMContentLoaded', () => {
    const launchBtn = document.getElementById('launch-btn');
    const overlay = document.getElementById('launch-overlay');

    launchBtn.addEventListener('click', async () => {
        // Show overlay
        overlay.classList.remove('hidden');

        try {
            // Call the python backend to launch the app
            const response = await fetch('/launch', {
                method: 'POST'
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('App launched successfully');
                // Keep overlay for a bit then hide
                setTimeout(() => {
                    overlay.classList.add('hidden');
                }, 3000);
            } else {
                alert('Error launching application: ' + data.message);
                overlay.classList.add('hidden');
            }
        } catch (error) {
            console.error('Failed to launch:', error);
            alert('Could not connect to the launch server. Make sure launch_portal.py is running.');
            overlay.classList.add('hidden');
        }
    });
});
