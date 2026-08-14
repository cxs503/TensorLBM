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
            <span>数据生产</span>
          </template>
          <el-menu-item index="/production/dashboard">总览</el-menu-item>
          <el-menu-item index="/production/cad">CAD 建模</el-menu-item>
          <el-menu-item index="/production/preprocess">预处理</el-menu-item>
          <el-menu-item index="/production/solve">求解</el-menu-item>
          <el-menu-item index="/production/postprocess">后处理</el-menu-item>
          <el-menu-item index="/production/benchmarks">基准</el-menu-item>
        </el-sub-menu>

        <el-sub-menu index="ai4s">
          <template #title>
            <el-icon><MagicStick /></el-icon>
            <span>AI4S 应用</span>
          </template>
          <el-menu-item index="/ai4s/apps">应用列表</el-menu-item>
          <el-menu-item index="/ai4s/run">运行应用</el-menu-item>
          <el-menu-item index="/ai4s/lineage">血缘追溯</el-menu-item>
        </el-sub-menu>

        <el-menu-item index="/data">
          <el-icon><Coin /></el-icon>
          <span>数据管理</span>
        </el-menu-item>

        <el-sub-menu index="misc">
          <template #title>
            <el-icon><Folder /></el-icon>
            <span>项目</span>
          </template>
          <el-menu-item index="/misc/projects">项目列表</el-menu-item>
          <el-menu-item index="/misc/reports">报告</el-menu-item>
        </el-sub-menu>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="app-header">
        <span class="header-title">{{ currentTitle }}</span>
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

const route = useRoute()
const activeMenu = computed(() => route.path)
const currentTitle = computed(() => (route.meta.title as string) || 'TensorLBM')
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
.app-main {
  background: #f0f2f5;
}
</style>
