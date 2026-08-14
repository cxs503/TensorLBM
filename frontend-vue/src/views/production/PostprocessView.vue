<template>
  <div>
    <el-card>
      <template #header>
        <div class="card-header">
          <span>后处理</span>
          <el-select
            v-model="selectedJobId"
            placeholder="选择作业"
            filterable
            style="width: 360px"
            @change="onJobChange"
          >
            <el-option
              v-for="j in selectableJobs"
              :key="j.job_id"
              :label="`${j.name} (${j.job_id})`"
              :value="j.job_id"
            />
          </el-select>
        </div>
      </template>

      <template v-if="!selectedJobId">
        <el-empty description="请先选择一个已完成的作业" />
      </template>

      <template v-else>
        <!-- 作业摘要 -->
        <el-descriptions v-if="summary" :column="4" border size="small" class="summary-block">
          <el-descriptions-item label="作业名称">{{ summary.job_name }}</el-descriptions-item>
          <el-descriptions-item label="类型">{{ summary.job_type }}</el-descriptions-item>
          <el-descriptions-item label="状态">{{ summary.status }}</el-descriptions-item>
          <el-descriptions-item label="耗时">{{ summary.duration_s ?? '—' }}s</el-descriptions-item>
          <el-descriptions-item label="PNG 快照">{{ summary.png_files }}</el-descriptions-item>
          <el-descriptions-item label="CSV 文件">{{ summary.csv_files }}</el-descriptions-item>
        </el-descriptions>

        <el-tabs v-model="activeTab" class="pp-tabs">
          <!-- 快照 -->
          <el-tab-pane label="快照" name="snapshots">
            <div v-loading="snapshotsLoading" class="snap-grid">
              <el-empty v-if="!snapshots.length && !snapshotsLoading" description="暂无快照" />
              <div v-for="img in snapshots" :key="img" class="snap-item">
                <el-image
                  :src="jobFileUrl(selectedJobId, img)"
                  :preview-src-list="snapshots.map((s) => jobFileUrl(selectedJobId, s))"
                  fit="contain"
                  lazy
                  class="snap-img"
                />
                <div class="snap-label">{{ img }}</div>
              </div>
            </div>
          </el-tab-pane>

          <!-- 云图查看器 -->
          <el-tab-pane label="云图查看器" name="viewer">
            <div class="viewer-toolbar">
              <el-select v-model="viewer.field" size="small" style="width: 200px">
                <el-option label="速度模" value="velocity_magnitude" />
                <el-option label="涡量" value="vorticity" />
                <el-option label="密度" value="density" />
                <el-option label="压力系数" value="pressure_coeff" />
                <el-option label="ux" value="ux" />
                <el-option label="uy" value="uy" />
              </el-select>
              <el-select v-model="viewer.checkpoint" size="small" style="width: 260px">
                <el-option label="latest" value="latest" />
                <el-option v-for="c in checkpoints" :key="c" :label="c" :value="c" />
              </el-select>
              <el-select v-model="viewer.colormap" size="small" style="width: 140px">
                <el-option v-for="(_, name) in cmaps" :key="name" :label="name" :value="name" />
              </el-select>
              <el-checkbox v-model="viewer.showArrows" size="small">矢量箭头</el-checkbox>
              <el-button size="small" type="primary" :loading="viewerLoading" @click="renderField">渲染</el-button>
            </div>
            <div class="viewer-body">
              <canvas ref="fieldCanvas" class="field-canvas"></canvas>
              <canvas ref="legendCanvas" class="legend-canvas" width="40" height="220"></canvas>
            </div>
            <div class="viewer-stats">{{ viewerStats }}</div>
          </el-tab-pane>

          <!-- 收敛曲线 -->
          <el-tab-pane label="收敛曲线" name="convergence">
            <el-button size="small" type="primary" :loading="convLoading" @click="loadConvergence">加载收敛数据</el-button>
            <el-table v-if="convRows.length" :data="convRows" size="small" border class="conv-table" max-height="400">
              <el-table-column prop="step" label="step" width="120" />
              <el-table-column
                v-for="col in convColumns"
                :key="col"
                :prop="col"
                :label="col"
              />
            </el-table>
            <el-empty v-else-if="convLoaded" description="暂无收敛数据" :image-size="60" />
          </el-tab-pane>

          <!-- 输出文件 -->
          <el-tab-pane label="输出文件" name="files">
            <el-table :data="files" v-loading="filesLoading" size="small" border>
              <el-table-column prop="path" label="路径" min-width="260" />
              <el-table-column label="大小" width="120">
                <template #default="{ row }">{{ (row.size / 1024).toFixed(1) }} KB</template>
              </el-table-column>
              <el-table-column prop="mime" label="类型" width="180" />
              <el-table-column label="操作" width="100">
                <template #default="{ row }">
                  <el-button size="small" link type="primary" @click="downloadFile(row.path)">下载</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>

          <!-- 运行日志 -->
          <el-tab-pane label="运行日志" name="logs">
            <pre class="log-box">{{ logsText }}</pre>
          </el-tab-pane>

          <!-- 元数据 -->
          <el-tab-pane label="元数据" name="metadata">
            <pre class="log-box">{{ metadataText }}</pre>
          </el-tab-pane>
        </el-tabs>
      </template>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import {
  getJobFiles,
  getJobImages,
  getJobLogs,
  getJobMetadata,
  jobFileUrl,
  listJobs,
  postCheckpoints,
  postConvergence,
  postFieldData,
  postSummary,
  type FieldDataResponse,
  type Job,
  type JobFile,
} from '@/api/production'

const route = useRoute()

const allJobs = ref<Job[]>([])
const selectedJobId = ref('')
const summary = ref<Record<string, any> | null>(null)
const activeTab = ref('snapshots')

const selectableJobs = computed(() =>
  allJobs.value.filter((j) => j.status === 'completed' || j.status === 'failed'),
)

// 快照
const snapshots = ref<string[]>([])
const snapshotsLoading = ref(false)

// 云图查看器
const viewer = ref({
  field: 'velocity_magnitude',
  checkpoint: 'latest',
  colormap: 'viridis',
  showArrows: false,
})
const checkpoints = ref<string[]>([])
const viewerLoading = ref(false)
const viewerStats = ref('')
const fieldCanvas = ref<HTMLCanvasElement | null>(null)
const legendCanvas = ref<HTMLCanvasElement | null>(null)

// 收敛
const convLoading = ref(false)
const convLoaded = ref(false)
const convRows = ref<Record<string, any>[]>([])
const convColumns = ref<string[]>([])

// 文件
const files = ref<JobFile[]>([])
const filesLoading = ref(false)

// 日志 / 元数据
const logsText = ref('')
const metadataText = ref('')

// ---------------------------------------------------------------------------
// 数据加载
// ---------------------------------------------------------------------------

async function loadJobs() {
  const res = await listJobs({ limit: 0 })
  allJobs.value = res.jobs
}

async function onJobChange() {
  if (!selectedJobId.value) return
  summary.value = null
  snapshots.value = []
  checkpoints.value = []
  files.value = []
  logsText.value = ''
  metadataText.value = ''
  convLoaded.value = false
  convRows.value = []
  convColumns.value = []

  await Promise.all([loadSummary(), loadSnapshots()])
  loadCheckpoints().catch(() => {})
  loadLogs()
  loadMetadata()
  loadFiles()
}

async function loadSummary() {
  try {
    summary.value = await postSummary(selectedJobId.value)
  } catch {
    // 摘要失败不阻塞其他数据
    const job = allJobs.value.find((j) => j.job_id === selectedJobId.value)
    if (job) {
      summary.value = {
        job_id: job.job_id,
        job_name: job.name,
        job_type: job.job_type,
        status: job.status,
        duration_s: job.total_duration_seconds,
        png_files: 0,
        csv_files: 0,
      }
    }
  }
}

async function loadSnapshots() {
  snapshotsLoading.value = true
  try {
    const r = await getJobImages(selectedJobId.value)
    snapshots.value = r.images
  } finally {
    snapshotsLoading.value = false
  }
}

async function loadCheckpoints() {
  const r = await postCheckpoints(selectedJobId.value)
  checkpoints.value = r.checkpoints
}

async function loadLogs() {
  try {
    const r = await getJobLogs(selectedJobId.value)
    logsText.value = r.logs.join('\n')
  } catch {
    logsText.value = ''
  }
}

async function loadMetadata() {
  try {
    const r = await getJobMetadata(selectedJobId.value)
    metadataText.value = JSON.stringify(r.metadata, null, 2)
  } catch {
    metadataText.value = ''
  }
}

async function loadFiles() {
  filesLoading.value = true
  try {
    const r = await getJobFiles(selectedJobId.value)
    files.value = r.files
  } finally {
    filesLoading.value = false
  }
}

async function loadConvergence() {
  convLoading.value = true
  try {
    const r = await postConvergence(selectedJobId.value)
    const series = (r.series as Record<string, any[]>) || {}
    const steps = (r.steps as number[]) || []
    convColumns.value = Object.keys(series)
    convRows.value = steps.map((s, i) => {
      const row: Record<string, any> = { step: s }
      for (const k of Object.keys(series)) {
        row[k] = series[k][i]
      }
      return row
    })
    convLoaded.value = true
  } finally {
    convLoading.value = false
  }
}

function downloadFile(path: string) {
  const a = document.createElement('a')
  a.href = jobFileUrl(selectedJobId.value, path)
  a.download = path.split('/').pop() || path
  a.click()
}

// ---------------------------------------------------------------------------
// 云图渲染（colormap + 矢量箭头）
// ---------------------------------------------------------------------------

interface RGB {
  r: number
  g: number
  b: number
}

type CmapStops = [number, number, number][]

const cmaps: Record<string, CmapStops> = {
  viridis: [
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.459], [0.254, 0.265, 0.53],
    [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.566, 0.551],
    [0.135, 0.659, 0.517], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.15], [0.993, 0.906, 0.144],
  ],
  plasma: [
    [0.05, 0.03, 0.528], [0.296, 0.008, 0.624], [0.494, 0.013, 0.657],
    [0.665, 0.064, 0.628], [0.807, 0.163, 0.548], [0.912, 0.286, 0.426],
    [0.973, 0.421, 0.303], [0.996, 0.564, 0.188], [0.981, 0.716, 0.147],
    [0.937, 0.875, 0.287], [0.94, 0.975, 0.131],
  ],
  hot: [
    [0, 0, 0], [0.333, 0, 0], [0.667, 0, 0], [1, 0, 0],
    [1, 0.333, 0], [1, 0.667, 0], [1, 1, 0], [1, 1, 0.5], [1, 1, 1],
  ],
  cool: [
    [0, 1, 1], [0.125, 0.875, 1], [0.25, 0.75, 1], [0.375, 0.625, 1],
    [0.5, 0.5, 1], [0.625, 0.375, 1], [0.75, 0.25, 1], [0.875, 0.125, 1], [1, 0, 1],
  ],
  rdbu: [
    [0.647, 0.082, 0.094], [0.839, 0.376, 0.302], [0.957, 0.647, 0.51],
    [0.992, 0.859, 0.78], [0.969, 0.969, 0.969], [0.82, 0.898, 0.941],
    [0.573, 0.773, 0.871], [0.263, 0.576, 0.765], [0.129, 0.4, 0.675],
  ],
}

function buildLut(stops: CmapStops): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3)
  const n = stops.length - 1
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    const s = t * n
    const lo = Math.floor(s)
    const hi = Math.min(lo + 1, n)
    const f = s - lo
    lut[i * 3] = Math.round((stops[lo][0] + f * (stops[hi][0] - stops[lo][0])) * 255)
    lut[i * 3 + 1] = Math.round((stops[lo][1] + f * (stops[hi][1] - stops[lo][1])) * 255)
    lut[i * 3 + 2] = Math.round((stops[lo][2] + f * (stops[hi][2] - stops[lo][2])) * 255)
  }
  return lut
}

function cmapColor(lut: Uint8ClampedArray, t: number): RGB {
  const i = Math.max(0, Math.min(255, Math.round(t * 255)))
  return { r: lut[i * 3], g: lut[i * 3 + 1], b: lut[i * 3 + 2] }
}

async function renderField() {
  if (!selectedJobId.value) return
  viewerLoading.value = true
  try {
    const r = await postFieldData(selectedJobId.value, {
      field: viewer.value.field,
      checkpoint: viewer.value.checkpoint,
    })
    await nextTick()
    drawField(r)
  } finally {
    viewerLoading.value = false
  }
}

function drawField(r: FieldDataResponse) {
  const canvas = fieldCanvas.value
  const legend = legendCanvas.value
  if (!canvas) return

  const lut = buildLut(cmaps[viewer.value.colormap] || cmaps.viridis)
  const nx = r.nx
  const ny = r.ny
  const fmin = r.field_min
  const fmax = r.field_max
  const range = fmax - fmin || 1

  const scale = Math.max(1, Math.min(6, Math.floor(700 / nx)))
  const cw = nx * scale
  const ch = ny * scale
  canvas.width = cw
  canvas.height = ch
  canvas.style.width = `${cw}px`
  canvas.style.height = `${ch}px`

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const img = ctx.createImageData(cw, ch)
  const pix = img.data
  for (let row = 0; row < ny; row++) {
    for (let col = 0; col < nx; col++) {
      const v = r.data[row * nx + col]
      const t = (v - fmin) / range
      const c = cmapColor(lut, t)
      for (let sy = 0; sy < scale; sy++) {
        for (let sx = 0; sx < scale; sx++) {
          const px = ((row * scale + sy) * cw + col * scale + sx) * 4
          pix[px] = c.r
          pix[px + 1] = c.g
          pix[px + 2] = c.b
          pix[px + 3] = 255
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0)

  // 矢量箭头
  if (viewer.value.showArrows && r.ux.length && r.uy.length) {
    const step = Math.max(4, Math.round(Math.max(nx, ny) / 18))
    let maxU = 0
    for (let i = 0; i < r.ux.length; i++) {
      const mag = Math.sqrt(r.ux[i] * r.ux[i] + r.uy[i] * r.uy[i])
      if (mag > maxU) maxU = mag
    }
    maxU = maxU || 1
    ctx.strokeStyle = 'rgba(255,255,255,0.8)'
    ctx.lineWidth = 1
    for (let row = step; row < ny - step / 2; row += step) {
      for (let col = step; col < nx - step / 2; col += step) {
        const idx = row * nx + col
        const ux = r.ux[idx]
        const uy = r.uy[idx]
        const mag = Math.sqrt(ux * ux + uy * uy)
        if (mag < 1e-10) continue
        const len = (mag / maxU) * step * scale * 0.85
        const cx0 = (col + 0.5) * scale
        const cy0 = (row + 0.5) * scale
        const dx = (ux / mag) * len
        const dy = (uy / mag) * len
        ctx.beginPath()
        ctx.moveTo(cx0 - dx / 2, cy0 - dy / 2)
        ctx.lineTo(cx0 + dx / 2, cy0 + dy / 2)
        ctx.stroke()
      }
    }
  }

  // 颜色条图例
  if (legend) {
    const lctx = legend.getContext('2d')
    if (lctx) {
      const h = legend.height
      for (let i = 0; i < h; i++) {
        const t = 1 - i / (h - 1)
        const c = cmapColor(lut, t)
        lctx.fillStyle = `rgb(${c.r},${c.g},${c.b})`
        lctx.fillRect(0, i, legend.width, 1)
      }
    }
  }

  viewerStats.value =
    `step: ${r.step}  |  网格: ${r.nx_orig}×${r.ny_orig}  |  ` +
    `min: ${fmin.toExponential(3)}  max: ${fmax.toExponential(3)}`
}

// ---------------------------------------------------------------------------
// 生命周期
// ---------------------------------------------------------------------------

watch(activeTab, (tab) => {
  if (tab === 'files') loadFiles()
  else if (tab === 'logs') loadLogs()
  else if (tab === 'metadata') loadMetadata()
})

onMounted(async () => {
  await loadJobs()
  const qid = route.query.job_id as string | undefined
  if (qid) {
    selectedJobId.value = qid
    await onJobChange()
  } else if (selectableJobs.value.length) {
    selectedJobId.value = selectableJobs.value[0].job_id
    await onJobChange()
  }
})
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.summary-block {
  margin-bottom: 16px;
}
.pp-tabs {
  margin-top: 8px;
}
.snap-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  min-height: 120px;
}
.snap-item {
  width: 200px;
}
.snap-img {
  width: 200px;
  height: 150px;
  border-radius: 4px;
  border: 1px solid #e4e7ed;
}
.snap-label {
  font-size: 11px;
  color: #909399;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-top: 4px;
}
.viewer-toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.viewer-body {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}
.field-canvas {
  border: 1px solid #e4e7ed;
  image-rendering: pixelated;
}
.legend-canvas {
  border: 1px solid #e4e7ed;
}
.viewer-stats {
  margin-top: 8px;
  color: #606266;
  font-size: 13px;
}
.conv-table {
  margin-top: 12px;
}
.log-box {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 12px;
  border-radius: 4px;
  max-height: 480px;
  overflow: auto;
  font-size: 12px;
  white-space: pre-wrap;
  margin: 0;
}
</style>
