<template>
  <div class="app-lineage">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>最近运行血缘链</span>
          <el-tag v-if="lastReport" type="success" effect="plain">
            {{ lastReport.name }} · {{ lastReport.family }}
          </el-tag>
        </div>
      </template>

      <el-alert
        class="hint-alert"
        type="info"
        :closable="false"
        show-icon
        title="血缘链来自最近一次本地全栈运行的 RunReport：lineage_upstream → 数据资产 → 数据集 → 任务 → 模型"
      />

      <template v-if="lastReport">
        <el-steps direction="vertical" :active="lineageSteps.length" class="lineage-steps">
          <el-step
            v-for="(step, index) in lineageSteps"
            :key="index"
            :title="step.title"
            :description="step.description"
          />
        </el-steps>
      </template>
      <el-empty
        v-else
        description="暂无血缘数据，请先在『运行应用』页完成一次本地全栈运行"
      />
    </el-card>

    <el-card class="query-card">
      <template #header>
        <span>按 job_id 查询运行状态</span>
      </template>

      <el-form inline>
        <el-form-item label="应用">
          <el-input
            v-model="queryApp"
            placeholder="应用名称，如 suboff_surrogate"
            style="width: 240px"
          />
        </el-form-item>
        <el-form-item label="job_id">
          <el-input v-model="queryJobId" placeholder="运行返回的 job_id" style="width: 240px" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="querying" @click="query">查询</el-button>
        </el-form-item>
      </el-form>

      <el-descriptions v-if="statusResult" :column="2" border>
        <el-descriptions-item label="job_id">{{ statusResult.job_id }}</el-descriptions-item>
        <el-descriptions-item label="app_name">{{ statusResult.app_name }}</el-descriptions-item>
        <el-descriptions-item label="status">{{ statusResult.status }}</el-descriptions-item>
        <el-descriptions-item label="scheduler_state">{{ statusResult.scheduler_state ?? '-' }}</el-descriptions-item>
        <el-descriptions-item label="hpc_job_id">{{ statusResult.hpc_job_id ?? '-' }}</el-descriptions-item>
        <el-descriptions-item label="elapsed">{{ statusResult.elapsed ?? '-' }}</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { getRunStatus } from '@/api/apps'
import type { RunStatusResponse } from '@/api/apps'
import { useRunHistory } from '@/stores/ai4sRun'

const { lastReport } = useRunHistory()

const queryApp = ref('')
const queryJobId = ref('')
const querying = ref(false)
const statusResult = ref<RunStatusResponse | null>(null)

interface LineageStep {
  title: string
  description: string
}

/** 由 RunReport 组装血缘链：lineage_upstream → data → dataset → job → model。 */
const lineageSteps = computed<LineageStep[]>(() => {
  const report = lastReport.value
  if (!report) return []
  const steps: LineageStep[] = (report.lineage_upstream ?? []).map((up) => ({
    title: '上游数据',
    description: up,
  }))
  steps.push({ title: '数据资产', description: report.data_asset_id })
  steps.push({ title: '数据集', description: report.dataset_asset_id })
  steps.push({ title: '任务', description: report.job_id })
  steps.push({ title: '模型', description: String(report.model_id) })
  return steps
})

async function query() {
  const app = queryApp.value.trim()
  const jobId = queryJobId.value.trim()
  if (!app) {
    ElMessage.warning('请输入应用名称')
    return
  }
  if (!jobId) {
    ElMessage.warning('请输入 job_id')
    return
  }
  querying.value = true
  try {
    statusResult.value = await getRunStatus(app, jobId)
  } catch {
    // 错误提示已由请求拦截器统一处理
  } finally {
    querying.value = false
  }
}
</script>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.hint-alert {
  margin-bottom: 16px;
}
.lineage-steps {
  padding: 8px 0;
}
.lineage-steps :deep(.el-step__description) {
  font-family: monospace;
  word-break: break-all;
}
.query-card {
  margin-top: 16px;
}
</style>
