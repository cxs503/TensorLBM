<template>
  <div class="apps-view">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>AI4S 应用列表</span>
          <el-button :icon="Refresh" :loading="loading" @click="fetchApps">刷新</el-button>
        </div>
      </template>

      <el-alert
        class="summary-alert"
        type="info"
        :closable="false"
        show-icon
        :title="`共发现 ${total} 个已注册的 AI4S 应用`"
      />

      <el-table v-loading="loading" :data="apps" stripe border style="width: 100%">
        <el-table-column type="index" label="#" width="60" align="center" />
        <el-table-column prop="name" label="应用名称" min-width="220" show-overflow-tooltip />
        <el-table-column prop="family" label="算法家族" min-width="180" show-overflow-tooltip>
          <template #default="{ row }">
            <el-tag type="success" effect="plain">{{ row.family }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="version" label="版本" width="120" align="center" />
        <el-table-column label="操作" width="140" align="center" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" size="small" :icon="VideoPlay" @click="goRun(row.name)">
              运行
            </el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty description="暂无应用，请确认后端 /api/apps 已就绪" />
        </template>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { Refresh, VideoPlay } from '@element-plus/icons-vue'
import { listApps } from '@/api/apps'
import type { AppInfo } from '@/api/apps'

const router = useRouter()

const loading = ref(false)
const apps = ref<AppInfo[]>([])
const total = ref(0)

async function fetchApps() {
  loading.value = true
  try {
    const res = await listApps()
    apps.value = res.apps ?? []
    total.value = res.total ?? res.apps.length
  } catch {
    // 错误提示已由 request 拦截器统一处理
  } finally {
    loading.value = false
  }
}

function goRun(name: string) {
  router.push(`/ai4s/run/${encodeURIComponent(name)}`)
}

onMounted(fetchApps)
</script>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.summary-alert {
  margin-bottom: 16px;
}
</style>
