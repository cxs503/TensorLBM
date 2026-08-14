<template>
  <div>
    <!-- 概览统计卡片 -->
    <el-row :gutter="16" class="stats-row">
      <el-col :span="5" v-for="s in summaryCards" :key="s.key">
        <el-card class="stat-card" shadow="hover">
          <div class="stat-label">{{ s.label }}</div>
          <div class="stat-value" :style="{ color: s.color }">{{ s.value }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card>
      <template #header>
        <div class="card-header">
          <span>作业总览</span>
          <div class="header-actions">
            <el-select
              v-model="statusFilter"
              placeholder="全部状态"
              clearable
              style="width: 150px"
              @change="loadJobs"
            >
              <el-option label="排队中" value="queued" />
              <el-option label="运行中" value="running" />
              <el-option label="已完成" value="completed" />
              <el-option label="失败" value="failed" />
              <el-option label="已取消" value="cancelled" />
            </el-select>
            <el-switch v-model="autoRefresh" active-text="自动刷新" />
            <el-button type="primary" :icon="Refresh" :loading="loading" @click="loadJobs">
              刷新
            </el-button>
          </div>
        </div>
      </template>

      <el-table :data="jobs" v-loading="loading" stripe style="width: 100%">
        <el-table-column prop="name" label="名称" min-width="200" show-overflow-tooltip />
        <el-table-column prop="job_id" label="作业 ID" width="120">
          <template #default="{ row }">
            <el-text class="mono" size="small">{{ row.job_id }}</el-text>
          </template>
        </el-table-column>
        <el-table-column prop="job_type" label="类型" width="160" show-overflow-tooltip />
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)" disable-transitions>{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="资源" width="120">
          <template #default="{ row }">
            <span class="muted">{{ row.assigned_resource || '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="耗时" width="110">
          <template #default="{ row }">
            <span class="muted">{{ formatDuration(row.total_duration_seconds) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="创建时间" width="190">
          <template #default="{ row }">
            <span class="muted">{{ formatTime(row.created_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="160" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="['queued', 'running'].includes(row.status)"
              size="small"
              type="warning"
              @click="onCancel(row)"
            >
              取消
            </el-button>
            <el-button
              v-if="['completed', 'failed', 'cancelled'].includes(row.status)"
              size="small"
              type="danger"
              plain
              @click="onDelete(row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty description="暂无作业" />
        </template>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { cancelJob, deleteJob, listJobs, type Job } from '@/api/production'

const jobs = ref<Job[]>([])
const loading = ref(false)
const statusFilter = ref('')
const autoRefresh = ref(true)

let timer: ReturnType<typeof setInterval> | null = null

const summaryCards = computed(() => {
  const total = jobs.value.length
  const running = jobs.value.filter((j) => j.status === 'running').length
  const completed = jobs.value.filter((j) => j.status === 'completed').length
  const failed = jobs.value.filter((j) => j.status === 'failed').length
  const queued = jobs.value.filter((j) => j.status === 'queued').length
  return [
    { key: 'total', label: '作业总数', value: total, color: '#409eff' },
    { key: 'running', label: '运行中', value: running, color: '#e6a23c' },
    { key: 'queued', label: '排队中', value: queued, color: '#909399' },
    { key: 'completed', label: '已完成', value: completed, color: '#67c23a' },
    { key: 'failed', label: '失败', value: failed, color: '#f56c6c' },
  ]
})

function statusLabel(status: string): string {
  const map: Record<string, string> = {
    queued: '排队中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }
  return map[status] || status
}

function statusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    queued: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    cancelled: 'info',
  }
  return map[status] || 'info'
}

function formatDuration(seconds: number | null): string {
  if (seconds == null) return '—'
  if (seconds < 1) return '<1s'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}m${s}s`
}

function formatTime(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

async function loadJobs() {
  loading.value = true
  try {
    const res = await listJobs({
      status: statusFilter.value || undefined,
      limit: 0,
    })
    jobs.value = res.jobs
  } finally {
    loading.value = false
  }
}

async function onCancel(row: Job) {
  try {
    await ElMessageBox.confirm(`确认取消作业「${row.name}」？`, '提示', { type: 'warning' })
  } catch {
    return
  }
  await cancelJob(row.job_id)
  ElMessage.success('已发送取消请求')
  await loadJobs()
}

async function onDelete(row: Job) {
  try {
    await ElMessageBox.confirm(`确认删除作业「${row.name}」？此操作不可恢复。`, '警告', { type: 'error' })
  } catch {
    return
  }
  await deleteJob(row.job_id)
  ElMessage.success('作业已删除')
  await loadJobs()
}

function startAutoRefresh() {
  stopAutoRefresh()
  timer = setInterval(() => {
    loadJobs()
  }, 5000)
}

function stopAutoRefresh() {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

onMounted(() => {
  loadJobs()
})

watch(
  autoRefresh,
  (val) => {
    if (val) startAutoRefresh()
    else stopAutoRefresh()
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  stopAutoRefresh()
})
</script>

<style scoped>
.stats-row {
  margin-bottom: 16px;
}
.stat-card {
  text-align: center;
}
.stat-label {
  font-size: 13px;
  color: #909399;
}
.stat-value {
  font-size: 28px;
  font-weight: 700;
  margin-top: 6px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}
.mono {
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}
.muted {
  color: #909399;
  font-size: 13px;
}
</style>
