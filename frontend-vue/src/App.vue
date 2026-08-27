<template>
  <el-container class="app-container">
    <el-aside width="220px" class="app-aside">
      <div class="app-logo">
        <span class="logo-text">TensorLBM</span>
        <span class="logo-sub">AI4S Platform</span>
      </div>
      <el-menu
        :default-active="activeMenu"
        router
        class="app-menu"
        background-color="#001529"
        text-color="#bfcbd9"
        active-text-color="#409eff"
      >
        <el-sub-menu index="production">
          <template #title>
            <el-icon><Cpu /></el-icon>
            <span>{{ t('menu.production') }}</span>
          </template>
          <el-menu-item index="/production/dashboard">{{ t('menu.productionChildren.dashboard') }}</el-menu-item>
          <el-menu-item index="/production/cad">{{ t('menu.productionChildren.cad') }}</el-menu-item>
          <el-menu-item index="/production/preprocess">{{ t('menu.productionChildren.preprocess') }}</el-menu-item>
          <el-menu-item index="/production/solve">{{ t('menu.productionChildren.solve') }}</el-menu-item>
          <el-menu-item index="/production/postprocess">{{ t('menu.productionChildren.postprocess') }}</el-menu-item>
          <el-menu-item index="/production/benchmarks">{{ t('menu.productionChildren.benchmarks') }}</el-menu-item>
        </el-sub-menu>

        <el-sub-menu index="ai4s">
          <template #title>
            <el-icon><MagicStick /></el-icon>
            <span>{{ t('menu.ai4s') }}</span>
          </template>
          <el-menu-item index="/ai4s/apps">{{ t('menu.ai4sChildren.apps') }}</el-menu-item>
          <el-menu-item index="/ai4s/run">{{ t('menu.ai4sChildren.run') }}</el-menu-item>
          <el-menu-item index="/ai4s/lineage">{{ t('menu.ai4sChildren.lineage') }}</el-menu-item>
        </el-sub-menu>

        <el-menu-item index="/data">
          <el-icon><Coin /></el-icon>
          <span>{{ t('menu.data') }}</span>
        </el-menu-item>

        <el-sub-menu index="misc">
          <template #title>
            <el-icon><Folder /></el-icon>
            <span>{{ t('menu.misc') }}</span>
          </template>
          <el-menu-item index="/misc/projects">{{ t('menu.miscChildren.projects') }}</el-menu-item>
          <el-menu-item index="/misc/reports">{{ t('menu.miscChildren.reports') }}</el-menu-item>
        </el-sub-menu>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="app-header">
        <span class="header-title">{{ currentTitle }}</span>
        <el-button
          class="lang-toggle"
          text
          @click="toggleLocale"
        >
          {{ t('common.switchTo') }}
        </el-button>
      </el-header>
      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { setLocale } from './i18n'

const route = useRoute()
const activeMenu = computed(() => route.path)
const currentTitle = computed(() => (route.meta.title as string) || 'TensorLBM')

const { t, locale } = useI18n()

function toggleLocale() {
  setLocale(locale.value === 'zh' ? 'en' : 'zh')
}
</script>

<style scoped>
.app-container {
  height: 100vh;
}
.app-aside {
  background-color: #001529;
  display: flex;
  flex-direction: column;
}
.app-logo {
  height: 60px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 20px;
  color: #fff;
}
.logo-text {
  font-size: 18px;
  font-weight: 700;
}
.logo-sub {
  font-size: 12px;
  color: #bfcbd9;
}
.app-menu {
  border-right: none;
  flex: 1;
}
.app-header {
  display: flex;
  align-items: center;
  border-bottom: 1px solid #e4e7ed;
  background: #fff;
}
.header-title {
  font-size: 16px;
  font-weight: 600;
}
.lang-toggle {
  margin-left: auto;
}
.app-main {
  background: #f0f2f5;
}
</style>
