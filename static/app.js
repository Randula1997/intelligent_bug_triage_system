const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");
const bugDatasetForm = document.getElementById("bug-dataset-form");
const bugDatasetFileInput = document.getElementById("bug-dataset-file");
const bugDatasetOutput = document.getElementById("bug-dataset-output");
const bugDatasetSubmitButton = document.getElementById("bug-dataset-submit-button");
const uploadForm = document.getElementById("upload-form");
const recommendForm = document.getElementById("recommend-form");
const uploadOutput = document.getElementById("upload-output");
const recommendOutput = document.getElementById("recommend-output");
const organizationDataStatus = document.getElementById("organization-data-status");
const expertiseFileInput = document.getElementById("expertise-file");
const uploadSubmitButton = uploadForm.querySelector('button[type="submit"]');
const clearOrganizationDataButton = document.getElementById("clear-organization-data-button");
const confirmClearModal = document.getElementById("confirm-clear-modal");
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
    renderBugDatasetMessage("Uploading bug dataset for validation...", false);

    try {
        const payload = await uploadBugDataset(file);
        renderBugDatasetMessage(JSON.stringify(payload, null, 2), false);
        bugDatasetForm.reset();
    } catch (error) {
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

clearOrganizationDataButton.addEventListener("click", () => {
    confirmClearModal.classList.remove("hidden");
});

cancelClearButton.addEventListener("click", closeClearModal);

confirmClearModal.addEventListener("click", (event) => {
    if (event.target === confirmClearModal) {
        closeClearModal();
    }
});

confirmClearButton.addEventListener("click", async () => {
    setClearButtonState(true);
    renderUploadMessage("Deleting organization vector data...", false);

    try {
        const response = await fetch("/api/organization-data", {
            method: "DELETE",
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "Failed to delete organization data.");
        }

        closeClearModal();
        renderUploadMessage(JSON.stringify(payload, null, 2), false);
        recommendOutput.className = "results empty";
        recommendOutput.innerHTML = "<p>Organization data cleared. Upload expertise data to generate new recommendations.</p>";
        uploadProgress.classList.add("hidden");
        await refreshOrganizationDataState();
    } catch (error) {
        renderUploadMessage(error.message, true);
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
            throw new Error(payload.detail || "Could not read current vector database state.");
        }

        const vectorCount = Number(payload.vector_count || 0);
        const hasData = vectorCount > 0;
        uploadUiState.hasData = hasData;
        syncUploadControls();
        renderOrganizationDataStatus(payload, hasData);

        if (hasData) {
            renderUploadMessage(
                `Current collection ${payload.collection_name} already contains ${vectorCount} vectors. Clear organization data before uploading a new dataset.`,
                false,
            );
            return;
        }

        renderUploadMessage("No dataset uploaded yet.", false);
    } catch (error) {
        uploadUiState.hasData = false;
        syncUploadControls();
        renderOrganizationDataStatus(null, false, error.message || "Could not read current vector database state.");
    }
}

function renderOrganizationDataStatus(payload, hasData, errorMessage = "") {
    if (errorMessage) {
        organizationDataStatus.className = "data-status error";
        organizationDataStatus.innerHTML = `
            <strong>Current vector database state is unavailable.</strong>
            <p>${escapeHtml(errorMessage)}</p>
        `;
        return;
    }

    const vectorCount = Number(payload.vector_count || 0);
    organizationDataStatus.className = hasData ? "data-status populated" : "data-status empty";
    organizationDataStatus.innerHTML = `
        <strong>${hasData ? "Organization data is loaded" : "Vector database is empty"}</strong>
        <p>Collection: ${escapeHtml(payload.collection_name)}</p>
        <p>Stored vectors: ${vectorCount}</p>
        <p>Embedding model: ${escapeHtml(payload.embedding_model_name)}</p>
    `;
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
    bugDatasetSubmitButton.disabled = isBusy;
    bugDatasetFileInput.disabled = isBusy;
}

function syncUploadControls() {
    uploadSubmitButton.disabled = uploadUiState.isBusy || uploadUiState.hasData;
    expertiseFileInput.disabled = uploadUiState.isBusy || uploadUiState.hasData;
    clearOrganizationDataButton.disabled = uploadUiState.isBusy || !uploadUiState.hasData;
}

async function uploadBugDataset(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/bug-dataset/upload", {
        method: "POST",
        body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
        throw new Error(payload.detail || "Bug dataset upload failed.");
    }

    return payload;
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
}

function setClearButtonState(isBusy) {
    confirmClearButton.disabled = isBusy;
    cancelClearButton.disabled = isBusy;
    clearOrganizationDataButton.disabled = isBusy;
}

function renderRecommendations(data) {
    const recommendations = data.recommendations || [];
    // const classifier = data.classifier_predictions || [];

    if (!recommendations.length) {
        recommendOutput.className = "results empty";
        recommendOutput.innerHTML = "<p>No recommendations found. Upload bug and developer expertise data first.</p>";
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

    // const classifierBlock = classifier.length
    //     ? `
    //         <section class="classifier-block">
    //             <h3>Classifier top labels</h3>
    //             <ul>
    //                 ${classifier.map((item) => `<li>${escapeHtml(item.developer_name)} <span>${Number(item.classifier_score).toFixed(4)}</span></li>`).join("")}
    //             </ul>
    //         </section>
    //     `
    //     : "";

    recommendOutput.className = "results";
    recommendOutput.innerHTML = `
        <div class="query-text">
            <h3>Combined Query</h3>
            <pre>${escapeHtml(data.query_text)}</pre>
        </div>
        <div class="result-list">${cards}</div>
        ${"" /* classifierBlock */}
    `;
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}
