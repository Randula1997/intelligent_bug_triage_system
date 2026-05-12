const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");
const bugDatasetForm = document.getElementById("bug-dataset-form");
const bugDatasetFileInput = document.getElementById("bug-dataset-file");
const bugDatasetOutput = document.getElementById("bug-dataset-output");
const bugDatasetSubmitButton = document.getElementById("bug-dataset-submit-button");
const clearBugDatasetButton = document.getElementById("clear-bug-dataset-button");
const bugDatasetProgress = document.getElementById("bug-dataset-progress");
const bugDatasetProgressBar = document.getElementById("bug-dataset-progress-bar");
const bugDatasetProgressPhase = document.getElementById("bug-dataset-progress-phase");
const bugDatasetProgressValue = document.getElementById("bug-dataset-progress-value");
const bugDatasetProgressMessage = document.getElementById("bug-dataset-progress-message");
const bugDatasetProgressTrack = bugDatasetProgress.querySelector(".progress-track");
const uploadForm = document.getElementById("upload-form");
const recommendForm = document.getElementById("recommend-form");
const uploadOutput = document.getElementById("upload-output");
const recommendOutput = document.getElementById("recommend-output");
const bugDatasetStatus = document.getElementById("bug-dataset-status");
const developerExpertiseStatus = document.getElementById("developer-expertise-status");
const expertiseFileInput = document.getElementById("expertise-file");
const uploadSubmitButton = uploadForm.querySelector('button[type="submit"]');
const clearOrganizationDataButton = document.getElementById("clear-organization-data-button");
const confirmClearModal = document.getElementById("confirm-clear-modal");
const confirmClearTitle = document.getElementById("confirm-clear-title");
const confirmClearDescription = document.getElementById("confirm-clear-description");
const cancelClearButton = document.getElementById("cancel-clear-button");
const confirmClearButton = document.getElementById("confirm-clear-button");
const uploadProgress = document.getElementById("upload-progress");
const uploadProgressBar = document.getElementById("upload-progress-bar");
const uploadProgressPhase = document.getElementById("upload-progress-phase");
const uploadProgressValue = document.getElementById("upload-progress-value");
const uploadProgressMessage = document.getElementById("upload-progress-message");
const uploadProgressTrack = uploadProgress.querySelector(".progress-track");

const uploadUiState = {
    isBusy: false,
    hasData: false,
};

const bugDatasetUiState = {
    isBusy: false,
    hasCheckpoint: false,
};

const clearModalState = {
    action: null,
};

tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
        activateTab(button.dataset.tabTarget);
    });
});

refreshOrganizationDataState();

bugDatasetForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = bugDatasetFileInput.files[0];
    if (!file) {
        renderBugDatasetMessage("Select a bug dataset file before uploading.", true);
        return;
    }

    setBugDatasetControlsState(true);
    renderBugDatasetProgress(0, "Uploading dataset", "Uploading bug dataset to start fine-tuning.");
    renderBugDatasetMessage("Starting fine-tuning job...", false);

    try {
        const job = await createBugDatasetJob(file);
        const payload = await pollBugDatasetJob(job.job_id);
        renderBugDatasetProgress(100, "Completed", "Fine-tuned checkpoint saved and activated.");
        renderBugDatasetMessage(JSON.stringify(payload, null, 2), false);
        bugDatasetForm.reset();
        await refreshOrganizationDataState();
    } catch (error) {
        renderBugDatasetProgress(100, "Failed", error.message, true);
        renderBugDatasetMessage(error.message, true);
    } finally {
        setBugDatasetControlsState(false);
    }
});

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById("expertise-file");
    const file = fileInput.files[0];
    if (!file) {
        renderUploadMessage("Select a dataset file before uploading.", true);
        return;
    }

    renderUploadProgress(0, "Uploading dataset", "Uploading expertise data to the server.");
    renderUploadMessage("Starting upload job...", false);
    setUploadControlsState(true);

    try {
        const job = await createUploadJob(file);
        const result = await pollUploadJob(job.job_id);
        renderUploadProgress(100, "Completed", "Expertise data uploaded and stored successfully.");
        renderUploadMessage(JSON.stringify(result, null, 2), false);
        await refreshOrganizationDataState();
    } catch (error) {
        renderUploadProgress(100, "Failed", error.message, true);
        renderUploadMessage(error.message, true);
    } finally {
        setUploadControlsState(false);
    }
});

recommendForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = {
        bug_title: document.getElementById("bug-title").value,
        bug_description: document.getElementById("bug-description").value,
        k: Number.parseInt(document.getElementById("top-k").value, 10),
    };

    recommendOutput.className = "results";
    recommendOutput.innerHTML = "<p>Embedding query and searching Milvus...</p>";

    try {
        const response = await fetch("/api/recommend", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Recommendation request failed.");
        }
        renderRecommendations(data);
    } catch (error) {
        recommendOutput.className = "results error";
        recommendOutput.innerHTML = `<p>${error.message}</p>`;
    }
});

clearBugDatasetButton.addEventListener("click", () => {
    openClearModal({
        action: "bug-dataset-model",
        title: "Delete fine-tuned bug dataset model?",
        description:
            "This will permanently remove the active fine-tuned checkpoint created from uploaded bug data. The original base checkpoint will not be changed.",
        confirmLabel: "Delete Model",
    });
});

clearOrganizationDataButton.addEventListener("click", () => {
    openClearModal({
        action: "developer-expertise-data",
        title: "Delete developer expertise data?",
        description:
            "This will permanently remove all vectors stored in the current Milvus collection for developer expertise data. You will need to upload expertise data again before running recommendations.",
        confirmLabel: "Delete Data",
    });
});

cancelClearButton.addEventListener("click", closeClearModal);

confirmClearModal.addEventListener("click", (event) => {
    if (event.target === confirmClearModal) {
        closeClearModal();
    }
});

confirmClearButton.addEventListener("click", async () => {
    setClearButtonState(true);
    try {
        if (clearModalState.action === "bug-dataset-model") {
            await clearBugDatasetModel();
        } else if (clearModalState.action === "developer-expertise-data") {
            await clearDeveloperExpertiseData();
        } else {
            closeClearModal();
            return;
        }
        closeClearModal();
    } catch (error) {
        if (clearModalState.action === "bug-dataset-model") {
            renderBugDatasetMessage(error.message, true);
        } else {
            renderUploadMessage(error.message, true);
        }
    } finally {
        setClearButtonState(false);
    }
});

function renderUploadMessage(message, isError) {
    uploadOutput.className = isError ? "output error" : "output";
    uploadOutput.textContent = message;
}

function renderBugDatasetMessage(message, isError) {
    bugDatasetOutput.className = isError ? "output error" : "output";
    bugDatasetOutput.textContent = message;
}

function activateTab(tabId) {
    tabButtons.forEach((button) => {
        const isActive = button.dataset.tabTarget === tabId;
        button.classList.toggle("active", isActive);
        button.setAttribute("aria-selected", String(isActive));
    });

    tabPanels.forEach((panel) => {
        const isActive = panel.id === tabId;
        panel.classList.toggle("active", isActive);
        panel.classList.toggle("hidden", !isActive);
    });
}

async function refreshOrganizationDataState() {
    try {
        const response = await fetch("/api/health");
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.detail || "Could not read current dataset and checkpoint state.");
        }

        const vectorCount = Number(payload.vector_count || 0);
        const hasData = vectorCount > 0;
        const hasCheckpoint = Boolean(payload.classifier_enabled);
        uploadUiState.hasData = hasData;
        bugDatasetUiState.hasCheckpoint = hasCheckpoint;
        syncUploadControls();
        syncBugDatasetControls();
        renderBugDatasetStatus(payload);
        renderDeveloperExpertiseStatus(payload, hasData);

        if (hasData) {
            renderUploadMessage(
                `Current collection ${payload.collection_name} already contains ${vectorCount} vectors. Clear developer expertise data before uploading a new dataset.`,
                false,
            );
            return;
        }

        renderUploadMessage("No dataset uploaded yet.", false);
    } catch (error) {
        uploadUiState.hasData = false;
        bugDatasetUiState.hasCheckpoint = false;
        syncUploadControls();
        syncBugDatasetControls();
        renderBugDatasetStatus(null, error.message || "Could not read current dataset and checkpoint state.");
        renderDeveloperExpertiseStatus(null, false, error.message || "Could not read current dataset and checkpoint state.");
    }
}

function renderBugDatasetStatus(payload, errorMessage = "") {
    if (errorMessage) {
        bugDatasetStatus.className = "data-status error";
        bugDatasetStatus.innerHTML = `
            <strong>Current checkpoint state is unavailable.</strong>
            <p class="status-line"><span class="status-value">${escapeHtml(errorMessage)}</span></p>
        `;
        return;
    }

    const classifierEnabled = Boolean(payload.classifier_enabled);
    const classifierBaseCheckpoint = payload.classifier_base_checkpoint || "Not configured";
    const classifierActiveCheckpoint = payload.classifier_active_checkpoint || "No fine-tuned checkpoint active";
    bugDatasetStatus.className = classifierEnabled ? "data-status populated" : "data-status empty";
    bugDatasetStatus.innerHTML = `
        <strong>${classifierEnabled ? "Fine-tuned checkpoint is active" : "No fine-tuned checkpoint active"}</strong>
        <p class="status-line"><span class="status-label">Base checkpoint</span><span class="status-value">${escapeHtml(classifierBaseCheckpoint)}</span></p>
        <p class="status-line"><span class="status-label">Active fine-tuned checkpoint</span><span class="status-value">${escapeHtml(classifierActiveCheckpoint)}</span></p>
        <p class="status-line"><span class="status-label">Fine-tuned model available</span><span class="status-value">${classifierEnabled ? "Yes" : "No"}</span></p>
    `;
}

function renderDeveloperExpertiseStatus(payload, hasData, errorMessage = "") {
    if (errorMessage) {
        developerExpertiseStatus.className = "data-status error";
        developerExpertiseStatus.innerHTML = `
            <strong>Current vector database state is unavailable.</strong>
            <p class="status-line"><span class="status-value">${escapeHtml(errorMessage)}</span></p>
        `;
        return;
    }

    const vectorCount = Number(payload.vector_count || 0);
    developerExpertiseStatus.className = hasData ? "data-status populated" : "data-status empty";
    developerExpertiseStatus.innerHTML = `
        <strong>${hasData ? "Organization data is loaded" : "Vector database is empty"}</strong>
        <p class="status-line"><span class="status-label">Embedding model</span><span class="status-value">${escapeHtml(payload.embedding_model_name)}</span></p>
        <p class="status-line"><span class="status-label">Developer expertise collection</span><span class="status-value">${escapeHtml(payload.collection_name)}</span></p>
        <p class="status-line"><span class="status-label">Stored expertise vectors</span><span class="status-value">${vectorCount}</span></p>
    `;
}

function renderBugDatasetProgress(percent, phase, message, isError = false) {
    const safePercent = Math.max(0, Math.min(100, percent));

    bugDatasetProgress.classList.remove("hidden");
    bugDatasetProgress.classList.toggle("error", isError);
    bugDatasetProgressBar.style.width = `${safePercent}%`;
    bugDatasetProgressPhase.textContent = phase;
    bugDatasetProgressValue.textContent = `${Math.round(safePercent)}%`;
    bugDatasetProgressMessage.textContent = message;
    bugDatasetProgressTrack.setAttribute("aria-valuenow", String(Math.round(safePercent)));
}

function renderUploadProgress(percent, phase, message, isError = false) {
    const safePercent = Math.max(0, Math.min(100, percent));

    uploadProgress.classList.remove("hidden");
    uploadProgress.classList.toggle("error", isError);
    uploadProgressBar.style.width = `${safePercent}%`;
    uploadProgressPhase.textContent = phase;
    uploadProgressValue.textContent = `${Math.round(safePercent)}%`;
    uploadProgressMessage.textContent = message;
    uploadProgressTrack.setAttribute("aria-valuenow", String(Math.round(safePercent)));
}

function setUploadControlsState(isBusy) {
    uploadUiState.isBusy = isBusy;
    syncUploadControls();
}

function setBugDatasetControlsState(isBusy) {
    bugDatasetUiState.isBusy = isBusy;
    syncBugDatasetControls();
}

function syncBugDatasetControls() {
    bugDatasetSubmitButton.disabled = bugDatasetUiState.isBusy || bugDatasetUiState.hasCheckpoint;
    bugDatasetFileInput.disabled = bugDatasetUiState.isBusy || bugDatasetUiState.hasCheckpoint;
    clearBugDatasetButton.disabled = bugDatasetUiState.isBusy || !bugDatasetUiState.hasCheckpoint;
}

function syncUploadControls() {
    uploadSubmitButton.disabled = uploadUiState.isBusy || uploadUiState.hasData;
    expertiseFileInput.disabled = uploadUiState.isBusy || uploadUiState.hasData;
    clearOrganizationDataButton.disabled = uploadUiState.isBusy || !uploadUiState.hasData;
}

function createBugDatasetJob(file) {
    return new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append("file", file);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/bug-dataset/upload");

        xhr.upload.addEventListener("progress", (event) => {
            if (!event.lengthComputable) {
                return;
            }

            const uploadPercent = (event.loaded / event.total) * 20;
            renderBugDatasetProgress(
                uploadPercent,
                "Uploading dataset",
                `Uploaded ${Math.round((event.loaded / event.total) * 100)}% of the bug dataset file.`,
            );
        });

        xhr.addEventListener("load", () => {
            let payload = {};
            try {
                payload = JSON.parse(xhr.responseText || "{}");
            } catch {
                reject(new Error("Bug dataset upload failed with an invalid server response."));
                return;
            }

            if (xhr.status < 200 || xhr.status >= 300) {
                reject(new Error(payload.detail || payload.error || "Bug dataset upload failed."));
                return;
            }

            renderBugDatasetProgress(25, "Upload complete", "Bug dataset received. Fine-tuning the checkpoint.");
            resolve(payload);
        });

        xhr.addEventListener("error", () => {
            reject(new Error("Bug dataset request failed before it reached the server."));
        });

        xhr.send(formData);
    });
}

async function pollBugDatasetJob(jobId) {
    while (true) {
        const response = await fetch(`/api/bug-dataset/upload/jobs/${jobId}`);
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.detail || "Could not read bug dataset training progress.");
        }

        const overallPercent = 25 + (payload.progress_percent * 0.75);
        renderBugDatasetProgress(overallPercent, toTitleCase(payload.phase), payload.message, payload.status === "failed");

        if (payload.status === "completed") {
            return payload.result;
        }

        if (payload.status === "failed") {
            throw new Error(payload.error || payload.message || "Bug dataset fine-tuning failed.");
        }

        await delay(700);
    }
}

function createUploadJob(file) {
    return new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append("file", file);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/expertise/upload/jobs");

        xhr.upload.addEventListener("progress", (event) => {
            if (!event.lengthComputable) {
                return;
            }

            const uploadPercent = (event.loaded / event.total) * 25;
            renderUploadProgress(uploadPercent, "Uploading dataset", `Uploaded ${Math.round((event.loaded / event.total) * 100)}% of the dataset file.`);
        });

        xhr.addEventListener("load", () => {
            let payload = {};
            try {
                payload = JSON.parse(xhr.responseText || "{}");
            } catch {
                reject(new Error("Upload failed with an invalid server response."));
                return;
            }

            if (xhr.status < 200 || xhr.status >= 300) {
                reject(new Error(payload.detail || payload.error || "Upload failed."));
                return;
            }

            renderUploadProgress(30, "Upload complete", "Dataset received. Creating embeddings and storing vectors.");
            resolve(payload);
        });

        xhr.addEventListener("error", () => {
            reject(new Error("Upload request failed before the dataset reached the server."));
        });

        xhr.send(formData);
    });
}

async function pollUploadJob(jobId) {
    while (true) {
        const response = await fetch(`/api/expertise/upload/jobs/${jobId}`);
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.detail || "Could not read upload progress.");
        }

        const overallPercent = 30 + (payload.progress_percent * 0.7);
        renderUploadProgress(overallPercent, toTitleCase(payload.phase), payload.message, payload.status === "failed");

        if (payload.status === "completed") {
            return payload.result;
        }

        if (payload.status === "failed") {
            throw new Error(payload.error || payload.message || "Upload failed.");
        }

        await delay(700);
    }
}

function delay(milliseconds) {
    return new Promise((resolve) => {
        window.setTimeout(resolve, milliseconds);
    });
}

function toTitleCase(value) {
    return value.charAt(0).toUpperCase() + value.slice(1);
}

function closeClearModal() {
    confirmClearModal.classList.add("hidden");
    clearModalState.action = null;
}

function setClearButtonState(isBusy) {
    confirmClearButton.disabled = isBusy;
    cancelClearButton.disabled = isBusy;
    clearOrganizationDataButton.disabled = isBusy || uploadUiState.isBusy || !uploadUiState.hasData;
    clearBugDatasetButton.disabled = isBusy || bugDatasetUiState.isBusy || !bugDatasetUiState.hasCheckpoint;
}

function openClearModal({ action, title, description, confirmLabel }) {
    clearModalState.action = action;
    confirmClearTitle.textContent = title;
    confirmClearDescription.textContent = description;
    confirmClearButton.textContent = confirmLabel;
    confirmClearModal.classList.remove("hidden");
}

async function clearDeveloperExpertiseData() {
    renderUploadMessage("Deleting developer expertise data...", false);

    const response = await fetch("/api/organization-data", {
        method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.detail || "Failed to delete developer expertise data.");
    }

    renderUploadMessage(JSON.stringify(payload, null, 2), false);
    recommendOutput.className = "results empty";
    recommendOutput.innerHTML = "<p>Developer expertise data cleared. Upload expertise data to generate new recommendations.</p>";
    uploadProgress.classList.add("hidden");
    await refreshOrganizationDataState();
}

async function clearBugDatasetModel() {
    renderBugDatasetMessage("Deleting fine-tuned bug dataset model...", false);

    const response = await fetch("/api/bug-dataset/model", {
        method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.detail || "Failed to delete the fine-tuned bug dataset model.");
    }

    renderBugDatasetMessage(JSON.stringify(payload, null, 2), false);
    bugDatasetProgress.classList.add("hidden");
    await refreshOrganizationDataState();
}

function renderRecommendations(data) {
    const recommendations = data.recommendations || [];
    const modelRecommendations = data.model_recommendations || [];
    const classifierEnabled = Boolean(data.classifier_enabled);
    const activeCheckpoint = data.active_model_checkpoint || "No fine-tuned checkpoint active";

    if (!recommendations.length && !modelRecommendations.length) {
        recommendOutput.className = "results empty";
        recommendOutput.innerHTML = "<p>No recommendations found. Upload expertise data and fine-tune a bug dataset first.</p>";
        return;
    }

    const cards = recommendations.map((item, index) => `
        <article class="result-card">
            <div class="result-rank">${index + 1}</div>
            <div class="result-body">
                <div class="result-header">
                    <h3>${escapeHtml(item.developer_name)}</h3>
                    <span>Similarity ${Number(item.similarity_score).toFixed(4)}</span>
                </div>
                <p><strong>Final score:</strong> ${Number(item.final_score).toFixed(4)}</p>
                ${"" /* item.classifier_score !== null && item.classifier_score !== undefined ? `<p><strong>Classifier score:</strong> ${Number(item.classifier_score).toFixed(4)}</p>` : "" */}
                ${item.matched_bug_text ? `<p class="matched-text">${escapeHtml(item.matched_bug_text)}</p>` : ""}
            </div>
        </article>
    `).join("");

    const vectorBlock = `
        <section class="results-section">
            <div class="results-section-heading">
                <h3>Vector Similarity Recommendations</h3>
                <p>Milvus search over uploaded developer expertise embeddings.</p>
            </div>
            ${recommendations.length ? `<div class="result-list">${cards}</div>` : "<p class=\"results-note\">No vector recommendations available. Upload expertise data first.</p>"}
        </section>
    `;

    const modelItems = modelRecommendations.length
        ? `
            <ul>
                ${modelRecommendations.map((item, index) => `
                    <li>
                        <span>${index + 1}. ${escapeHtml(item.developer_name)}</span>
                        <span>${Number(item.model_score).toFixed(4)}</span>
                    </li>
                `).join("")}
            </ul>
        `
        : `<p class="results-note">${classifierEnabled ? "No model recommendations were returned for this query." : "No fine-tuned checkpoint is active yet. Upload a bug dataset to train one."}</p>`;

    const modelBlock = `
        <section class="classifier-block">
            <div class="results-section-heading">
                <h3>Fine-Tuned Model Recommendations</h3>
                <p>Predictions from the active checkpoint: ${escapeHtml(activeCheckpoint)}</p>
            </div>
            ${modelItems}
        </section>
    `;

    recommendOutput.className = "results";
    recommendOutput.innerHTML = `
        <div class="query-text">
            <h3>Combined Query</h3>
            <pre>${escapeHtml(data.query_text)}</pre>
        </div>
        ${vectorBlock}
        ${modelBlock}
    `;
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}
