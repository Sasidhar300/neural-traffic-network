// JavaScript for the simulation output page

document.addEventListener('DOMContentLoaded', function() {
    const consoleOutput = document.getElementById('console-output');
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const viewResultsBtn = document.getElementById('view-results-btn');
    const errorBtn = document.getElementById('error-btn');
    let isComplete = false;
    let isFirstSseMessage = true;

    if (consoleOutput && statusIndicator && statusText && viewResultsBtn) {
        // Connect to SSE endpoint
        const sseUrl = document.querySelector('meta[name="sse-url"]').getAttribute('content');
        const eventSource = new EventSource(sseUrl);

        // Handle incoming messages
        eventSource.onmessage = function(event) {
            if (event.data === "[SIMULATION_COMPLETE]") {
                // Simulation is complete
                isComplete = true;
                isFirstSseMessage = false; // Ensure clearing doesn't happen if completion is first message
                statusIndicator.classList.remove('status-running');
                statusIndicator.classList.add('status-complete');
                statusText.textContent = "Simulation complete!";

                // Show the "View Results" button
                viewResultsBtn.style.display = 'inline-block';
                eventSource.close();
            }
            else if (event.data === "[SIMULATION_ERROR]") {
                // Simulation encountered an error
                isComplete = true;
                isFirstSseMessage = false;
                statusIndicator.classList.remove('status-running');
                statusIndicator.style.backgroundColor = "#dc3545";
                statusText.textContent = "Simulation error!";

                // Hide the "View Results" button
                errorBtn.style.display = 'inline-block';
                eventSource.close();
            }
            else if (event.data.startsWith(':')) {
                return;
            }
            else {
                // Replace escaped newlines with <br> tags for proper HTML display
                let formattedOutput = event.data.replace(/\\n/g, '<br>');
                if (consoleOutput.innerHTML.length > 0 && !consoleOutput.innerHTML.endsWith('<br>')) {
                    consoleOutput.innerHTML += '<br>'; // Ensure newline before appending if needed
                }
                consoleOutput.innerHTML += formattedOutput;
                consoleOutput.scrollTop = consoleOutput.scrollHeight;
            }
        };

        eventSource.onerror = function() {
            if (!isComplete) {
                consoleOutput.innerHTML += "\nConnection to server lost. The simulation may still be running.\n";
                statusText.textContent = "Connection lost";
                statusIndicator.classList.remove('status-running');
                isComplete = true;
                isFirstSseMessage = false;
                errorBtn.style.display = 'inline-block';
                statusIndicator.style.backgroundColor = "#dc3545";
            } else {
                 console.error("SSE error after simulation completion.");
            }
            eventSource.close();
        };
    } else {
        console.error("One or more required elements (console-output, status-indicator, status-text, view-results-btn) not found.");
    }
});
