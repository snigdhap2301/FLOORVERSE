// State management
const state = {
  unit: "metric",
  floors: 1,
  totalArea: 100,
  houseIndex: 0,
  rooms: [
    { type: "Bedroom", count: 1 },
    { type: "Bathroom", count: 1 },
    { type: "Kitchen", count: 1 },
    { type: "Living Room", count: 1 },
  ],
  expandedSections: {
    unit: true,
    area: true,
    rooms: true,
    houseIndex: true,
  },
  isGenerating: false,
}

// Available room types
const availableRooms = ["Dining Room", "Office", "Storage Room", "Balcony"]

// DOM Elements
const appContainer = document.getElementById("appContainer")
const parametersPanel = document.getElementById("parametersPanel")
const previewPanel = document.getElementById("previewPanel")
const unitButtons = document.querySelectorAll(".unit-btn")
const totalAreaInput = document.getElementById("totalArea")
const houseIndexInput = document.getElementById("houseIndex")
const areaDisplay = document.getElementById("areaDisplay")
const roomsList = document.getElementById("roomsList")
const addRoomBtn = document.getElementById("addRoom")
const generateBtn = document.getElementById("generateBtn")
const closePreviewBtn = document.getElementById("closePreviewBtn")
const floorplanContainer = document.getElementById("floorplanContainer")
const loadingIndicator = document.getElementById("loadingIndicator")
const metricsContainer = document.getElementById("metricsContainer")
const metricsGrid = document.getElementById("metricsGrid")

// Toggle sections
const sections = [
  { id: "unit", header: "unitHeader", content: "unitContent" },
  { id: "area", header: "areaHeader", content: "areaContent" },
  { id: "rooms", header: "roomsHeader", content: "roomsContent" },
  { id: "houseIndex", header: "houseIndexHeader", content: "houseIndexContent" },
]

sections.forEach((section) => {
  const header = document.getElementById(section.header)
  const content = document.getElementById(section.content)
  const toggleIcon = header.querySelector(".toggle-icon")

  if (header && content && toggleIcon) {
    header.addEventListener("click", () => {
      state.expandedSections[section.id] = !state.expandedSections[section.id]
      content.style.display = state.expandedSections[section.id] ? "block" : "none"
      toggleIcon.classList.toggle("open", state.expandedSections[section.id])
    })
  }
})

// Helper Functions
function getAreaUnit() {
  return state.unit === "metric" ? "m²" : "ft²"
}

function updateAreaDisplay() {
  areaDisplay.textContent = `${state.totalArea} ${getAreaUnit()}`
}

function createRoomElement(room, index) {
  const div = document.createElement("div")
  div.className = "room-item"
  div.innerHTML = `
        <div class="room-type">${room.type}</div>
        <div class="room-controls">
            <button class="room-btn decrease" aria-label="Decrease ${room.type}" ${room.count === 0 ? "disabled" : ""}>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
            </button>
            <span class="room-count">${room.count}</span>
            <button class="room-btn increase" aria-label="Increase ${room.type}">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
            </button>
        </div>
    `

  // Add event listeners
  div.querySelector(".decrease").addEventListener("click", () => handleRoomCount(index, false))
  div.querySelector(".increase").addEventListener("click", () => handleRoomCount(index, true))

  return div
}

function updateRoomsList() {
  roomsList.innerHTML = ""
  state.rooms.forEach((room, index) => {
    roomsList.appendChild(createRoomElement(room, index))
  })

  // Show/hide add room button based on available rooms
  addRoomBtn.style.display = state.rooms.length >= availableRooms.length + 4 ? "none" : "flex"
}

function parseEfficiencyMetrics(imageUrl) {
  // This function would normally extract metrics from the image
  // For demo purposes, we'll create sample metrics
  return {
    "Space Utilization": 0.85,
    "Room Ratio": 0.92,
    "Circulation Efficiency": 0.78,
    "Window Placement": 0.88,
    "Door Accessibility": 0.95,
    "Overall Score": 0.87,
  }
}

function displayMetrics(metrics) {
  metricsGrid.innerHTML = ""

  Object.entries(metrics).forEach(([key, value]) => {
    const metricItem = document.createElement("div")
    metricItem.className = "metric-item"
    metricItem.innerHTML = `
            <div class="metric-label">${key}</div>
            <div class="metric-value">${(value * 100).toFixed(0)}%</div>
        `
    metricsGrid.appendChild(metricItem)
  })

  metricsContainer.classList.remove("hidden")
}

function setGeneratingState(isGenerating) {
  state.isGenerating = isGenerating

  if (isGenerating) {
    appContainer.classList.add("generating")
    previewPanel.classList.add("active")
    loadingIndicator.classList.remove("hidden")
    floorplanContainer.innerHTML = '<p class="text-center text-lg text-muted">Generating your floor plan...</p>'
    metricsContainer.classList.add("hidden")
    generateBtn.disabled = true
    generateBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="animate-spin">
                <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
            </svg>
            Generating...
        `
  } else {
    loadingIndicator.classList.add("hidden")
    generateBtn.disabled = false
    generateBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"></path>
                <path d="m9 12 2 2 4-4"></path>
            </svg>
            Generate Floor Plan
        `
  }
}

function resetPreview() {
  previewPanel.classList.remove("active")
  appContainer.classList.remove("generating")
  floorplanContainer.innerHTML = '<p class="text-center text-lg text-muted">Your floor plan will appear here</p>'
  metricsContainer.classList.add("hidden")
}

// Event Handlers
function handleUnitChange(event) {
  const selectedUnit = event.target.closest(".unit-btn").dataset.unit
  if (selectedUnit) {
    state.unit = selectedUnit
    unitButtons.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.unit === selectedUnit)
    })

    // Convert area if needed
    if (selectedUnit === "imperial" && state.unit === "metric") {
      state.totalArea = Math.round(state.totalArea * 10.764)
    } else if (selectedUnit === "metric" && state.unit === "imperial") {
      state.totalArea = Math.round(state.totalArea / 10.764)
    }

    totalAreaInput.value = state.totalArea
    updateAreaDisplay()
  }
}

function handleRoomCount(index, increment) {
  if (increment) {
    state.rooms[index].count++
  } else if (state.rooms[index].count > 0) {
    state.rooms[index].count--
  }
  updateRoomsList()
}

function addRoom() {
  const availableRoom = availableRooms.find((room) => !state.rooms.some((existing) => existing.type === room))

  if (availableRoom) {
    state.rooms.push({ type: availableRoom, count: 1 })
    updateRoomsList()
  }
}

async function generateFloorPlan() {
  try {
    setGeneratingState(true)

    const response = await fetch("/generate_floorplan", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        house_index: state.houseIndex,
        unit: state.unit,
        total_area: state.totalArea,
        rooms: state.rooms,
      }),
    })

    if (!response.ok) {
      throw new Error("Failed to generate floor plan")
    }

    // Convert the response to a blob (image)
    const blob = await response.blob()
    const imageUrl = URL.createObjectURL(blob)

    // Update preview
    floorplanContainer.innerHTML = `
            <img src="${imageUrl}" alt="Generated Floor Plan" class="floorplan-image" />
        `

    // Parse and display metrics
    const metrics = parseEfficiencyMetrics(imageUrl)
    displayMetrics(metrics)
  } catch (error) {
    floorplanContainer.innerHTML = `
            <div class="text-error p-4 text-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mb-4">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <p class="font-medium">Error: ${error.message}</p>
                <p class="mt-4 text-sm">Please check the server logs for more details.</p>
            </div>
        `
  } finally {
    setGeneratingState(false)
  }
}

// Initialize
function init() {
  // Set up event listeners
  unitButtons.forEach((btn) => btn.addEventListener("click", handleUnitChange))

  totalAreaInput.addEventListener("input", (e) => {
    state.totalArea = Number(e.target.value)
    updateAreaDisplay()
  })

  houseIndexInput.addEventListener("input", (e) => {
    state.houseIndex = Number(e.target.value)
  })

  addRoomBtn.addEventListener("click", addRoom)
  generateBtn.addEventListener("click", generateFloorPlan)
  closePreviewBtn.addEventListener("click", resetPreview)

  // Initial render
  updateRoomsList()
  updateAreaDisplay()
}

// Start the application
document.addEventListener("DOMContentLoaded", init)

