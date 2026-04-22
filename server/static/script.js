document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('video-input');
    const adInput = document.getElementById('ad-input');
    const videoZone = document.getElementById('video-zone');
    const adZone = document.getElementById('ad-zone');
    const renderBtn = document.getElementById('render-btn');
    const videoOutput = document.getElementById('video-output');
    const overlay = document.getElementById('processing-overlay');
    const progressText = document.getElementById('progress-text');

    let videoFile = null;
    let adFile = null;

    // --- Drag & Drop Handlers ---
    const setupDropzone = (zone, input, callback) => {
        zone.addEventListener('click', () => input.click());
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('active');
        });
        zone.addEventListener('dragleave', () => zone.classList.remove('active'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('active');
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                callback(e.dataTransfer.files[0]);
            }
        });
        input.addEventListener('change', () => {
            if (input.files.length) callback(input.files[0]);
        });
    };

    setupDropzone(videoZone, videoInput, (file) => {
        videoFile = file;
        videoZone.querySelector('.file-info').textContent = `Video: ${file.name}`;
        checkReady();
    });

    setupDropzone(adZone, adInput, (file) => {
        adFile = file;
        adZone.querySelector('.file-info').textContent = `Ad: ${file.name}`;
        checkReady();
    });

    // --- Slider UI Sync ---
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        const tag = slider.parentElement.querySelector('.value-tag');
        slider.addEventListener('input', () => {
            tag.textContent = slider.value + (slider.id === 'rotation' ? '°' : '');
        });
    });

    const checkReady = () => {
        renderBtn.disabled = !(videoFile && adFile);
    };

    // --- API Call ---
    renderBtn.addEventListener('click', async () => {
        if (!videoFile || !adFile) return;

        overlay.style.display = 'flex';
        renderBtn.disabled = true;
        progressText.textContent = "Analyzing scene & estimating depth...";

        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('ad', adFile);
        formData.append('scale', document.getElementById('scale').value);
        formData.append('rotation', document.getElementById('rotation').value);
        formData.append('tilt', document.getElementById('tilt').value);
        formData.append('alpha', document.getElementById('alpha').value);
        formData.append('feather', document.getElementById('feather').value);
        formData.append('shadow', document.getElementById('shadow').value);

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Processing failed');

            const result = await response.json();
            
            // Update Video
            videoOutput.style.display = 'none'; // Hide while loading
            videoOutput.src = result.video_url + '?t=' + new Date().getTime();
            videoOutput.load();
            
            videoOutput.oncanplay = () => {
                overlay.style.display = 'none';
                document.getElementById('output-placeholder').style.display = 'none';
                videoOutput.style.display = 'block';
                videoOutput.play();
            };

            // Update Metrics
            updateMetrics(result.metrics);

        } catch (error) {
            alert('Error: ' + error.message);
            overlay.style.display = 'none';
        } finally {
            renderBtn.disabled = false;
        }
    });

    const updateMetrics = (metrics) => {
        const updateMetric = (id, val, color) => {
            const card = document.getElementById(`metric-${id}`);
            card.querySelector('.metric-value').textContent = val.toFixed(2);
            const fill = card.querySelector('.progress-fill');
            fill.style.width = (val * 100) + '%';
            if (color) fill.style.background = color;
        };

        updateMetric('quality', metrics.total, '#3b82f6');
        updateMetric('stability', metrics.temporal, '#10b981');
        updateMetric('realism', metrics.realism, '#60a5fa');
        updateMetric('alignment', metrics.alignment, '#a855f7');
    };
});
