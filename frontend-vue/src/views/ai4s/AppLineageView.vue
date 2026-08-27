<template>
  <div class="app-lineage">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('ai4s.lineage.title') }}</span>
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
        :title="t('ai4s.lineage.hint')"
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
        :description="t('ai4s.lineage.empty')"
      />
    </el-card>

    <el-card class="query-card">
      <template #header>
        <span>{{ t('ai4s.lineage.queryTitle') }}</span>
      </template>

      <el-form inline>
        <el-form-item :label="t('ai4s.lineage.app')">
          <el-input
            v-model="queryApp"
            :placeholder="t('ai4s.lineage.appPlaceholder')"
            style="width: 240px"
          />
        </el-form-item>
        <el-form-item label="job_id">
          <el-input v-model="queryJobId" :placeholder="t('ai4s.lineage.jobIdPlaceholder')" style="width: 240px" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="querying" @click="query">{{ t('ai4s.lineage.query') }}</el-button>
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
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { getRunStatus } from '@/api/apps'
import type { RunStatusResponse } from '@/api/apps'
import { useRunHistory } from '@/stores/ai4sRun'

const { t } = useI18n()
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
    title: t('ai4s.lineage.steps.upstream'),
    description: up,
  }))
  steps.push({ title: t('ai4s.lineage.steps.dataAsset'), description: report.data_asset_id })
  steps.push({ title: t('ai4s.lineage.steps.dataset'), description: report.dataset_asset_id })
  steps.push({ title: t('ai4s.lineage.steps.job'), description: report.job_id })
  steps.push({ title: t('ai4s.lineage.steps.model'), description: String(report.model_id) })
  return steps
})

async function query() {
  const app = queryApp.value.trim()
  const jobId = queryJobId.value.trim()
  if (!app) {
    ElMessage.warning(t('ai4s.lineage.messages.enterAppName'))
    return
  }
  if (!jobId) {
    ElMessage.warning(t('ai4s.lineage.messages.enterJobId'))
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
