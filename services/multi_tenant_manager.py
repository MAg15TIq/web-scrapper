"""
Multi-Tenant Support System for Phase 3: Enterprise & Scalability Features.

This module provides tenant isolation, resource quotas, tenant-specific configurations,
and billing/usage tracking for enterprise multi-tenancy.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import hashlib

from services.redis_service import RedisService
from services.database_service import DatabaseService


class TenantStatus(Enum):
    """Tenant status enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    EXPIRED = "expired"
    PENDING = "pending"


class BillingModel(Enum):
    """Billing model enumeration."""
    PAY_PER_USE = "pay_per_use"
    SUBSCRIPTION = "subscription"
    FREEMIUM = "freemium"
    ENTERPRISE = "enterprise"


@dataclass
class ResourceQuota:
    """Resource quota configuration for a tenant."""
    max_concurrent_jobs: int = 10
    max_daily_requests: int = 1000
    max_monthly_requests: int = 30000
    max_cpu_hours: float = 100.0
    max_memory_gb_hours: float = 200.0
    max_storage_gb: float = 10.0
    max_bandwidth_gb: float = 100.0
    max_api_calls_per_minute: int = 60
    max_agents: int = 5
    priority_level: int = 1  # 1-10, higher is better


@dataclass
class TenantConfig:
    """Tenant-specific configuration."""
    tenant_id: str
    name: str
    email: str
    status: TenantStatus
    billing_model: BillingModel
    quota: ResourceQuota
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    allowed_domains: List[str] = None
    custom_settings: Dict[str, Any] = None


@dataclass
class UsageMetrics:
    """Usage metrics for a tenant."""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    total_requests: int = 0
    cpu_hours_used: float = 0.0
    memory_gb_hours_used: float = 0.0
    storage_gb_used: float = 0.0
    bandwidth_gb_used: float = 0.0
    api_calls: int = 0
    total_cost: float = 0.0


class MultiTenantManager:
    """
    Multi-tenant management system providing tenant isolation, resource quotas,
    billing tracking, and tenant-specific configurations.
    """
    
    def __init__(self, redis_service: RedisService, database_service: DatabaseService):
        self.redis_service = redis_service
        self.database_service = database_service
        self.logger = logging.getLogger("multi_tenant_manager")
        
        # Tenant management
        self.tenants: Dict[str, TenantConfig] = {}
        self.tenant_usage: Dict[str, UsageMetrics] = {}
        self.tenant_sessions: Dict[str, Set[str]] = defaultdict(set)
        
        # Resource tracking
        self.resource_allocations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.usage_history: Dict[str, List[UsageMetrics]] = defaultdict(list)
        
        # Rate limiting
        self.rate_limit_counters: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.rate_limit_windows: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        
        # Background tasks
        self.usage_tracking_task: Optional[asyncio.Task] = None
        self.quota_enforcement_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Pricing configuration
        self.pricing_config = {
            "cpu_hour_cost": 0.10,
            "memory_gb_hour_cost": 0.05,
            "storage_gb_cost": 0.02,
            "bandwidth_gb_cost": 0.01,
            "api_call_cost": 0.001,
            "job_base_cost": 0.05
        }
        
        self.logger.info("Multi-tenant manager initialized")
    
    async def start(self):
        """Start the multi-tenant manager."""
        if not self.is_running:
            self.is_running = True
            await self._load_tenants()
            
            # Start background tasks
            self.usage_tracking_task = asyncio.create_task(self._usage_tracking_loop())
            self.quota_enforcement_task = asyncio.create_task(self._quota_enforcement_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.logger.info("Multi-tenant manager started")
    
    async def stop(self):
        """Stop the multi-tenant manager."""
        if self.is_running:
            self.is_running = False
            
            # Stop background tasks
            tasks = [self.usage_tracking_task, self.quota_enforcement_task, self.cleanup_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            await self._save_tenants()
            self.logger.info("Multi-tenant manager stopped")
    
    async def create_tenant(
        self,
        name: str,
        email: str,
        billing_model: BillingModel = BillingModel.FREEMIUM,
        quota: ResourceQuota = None,
        expires_at: Optional[datetime] = None,
        metadata: Dict[str, Any] = None,
        allowed_domains: List[str] = None,
        custom_settings: Dict[str, Any] = None
    ) -> str:
        """Create a new tenant."""
        try:
            tenant_id = str(uuid.uuid4())
            
            # Set default quota based on billing model
            if quota is None:
                quota = self._get_default_quota(billing_model)
            
            # Create tenant configuration
            tenant_config = TenantConfig(
                tenant_id=tenant_id,
                name=name,
                email=email,
                status=TenantStatus.ACTIVE,
                billing_model=billing_model,
                quota=quota,
                created_at=datetime.now(),
                expires_at=expires_at,
                metadata=metadata or {},
                allowed_domains=allowed_domains or [],
                custom_settings=custom_settings or {}
            )
            
            # Store tenant
            self.tenants[tenant_id] = tenant_config
            
            # Initialize usage metrics
            self.tenant_usage[tenant_id] = UsageMetrics(
                tenant_id=tenant_id,
                period_start=datetime.now(),
                period_end=datetime.now() + timedelta(days=30)
            )
            
            # Save to database
            await self._save_tenant(tenant_config)
            
            self.logger.info(f"Created tenant: {tenant_id} ({name})")
            return tenant_id
            
        except Exception as e:
            self.logger.error(f"Failed to create tenant: {e}")
            raise
    
    async def update_tenant(
        self,
        tenant_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update tenant configuration."""
        try:
            if tenant_id not in self.tenants:
                self.logger.warning(f"Tenant {tenant_id} not found")
                return False
            
            tenant = self.tenants[tenant_id]
            
            # Update allowed fields
            allowed_updates = [
                'name', 'email', 'status', 'billing_model', 'quota',
                'expires_at', 'metadata', 'allowed_domains', 'custom_settings'
            ]
            
            for field, value in updates.items():
                if field in allowed_updates:
                    if field == 'status':
                        tenant.status = TenantStatus(value)
                    elif field == 'billing_model':
                        tenant.billing_model = BillingModel(value)
                    elif field == 'quota' and isinstance(value, dict):
                        # Update quota fields
                        for quota_field, quota_value in value.items():
                            if hasattr(tenant.quota, quota_field):
                                setattr(tenant.quota, quota_field, quota_value)
                    else:
                        setattr(tenant, field, value)
            
            # Save to database
            await self._save_tenant(tenant)
            
            self.logger.info(f"Updated tenant: {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update tenant {tenant_id}: {e}")
            return False
    
    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all associated data."""
        try:
            if tenant_id not in self.tenants:
                self.logger.warning(f"Tenant {tenant_id} not found")
                return False
            
            # Remove from memory
            self.tenants.pop(tenant_id, None)
            self.tenant_usage.pop(tenant_id, None)
            self.tenant_sessions.pop(tenant_id, None)
            self.resource_allocations.pop(tenant_id, None)
            self.usage_history.pop(tenant_id, None)
            self.rate_limit_counters.pop(tenant_id, None)
            self.rate_limit_windows.pop(tenant_id, None)
            
            # Remove from database
            await self._delete_tenant_from_db(tenant_id)
            
            self.logger.info(f"Deleted tenant: {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete tenant {tenant_id}: {e}")
            return False
    
    def _get_default_quota(self, billing_model: BillingModel) -> ResourceQuota:
        """Get default quota based on billing model."""
        if billing_model == BillingModel.FREEMIUM:
            return ResourceQuota(
                max_concurrent_jobs=2,
                max_daily_requests=100,
                max_monthly_requests=3000,
                max_cpu_hours=10.0,
                max_memory_gb_hours=20.0,
                max_storage_gb=1.0,
                max_bandwidth_gb=10.0,
                max_api_calls_per_minute=10,
                max_agents=1,
                priority_level=1
            )
        elif billing_model == BillingModel.SUBSCRIPTION:
            return ResourceQuota(
                max_concurrent_jobs=10,
                max_daily_requests=1000,
                max_monthly_requests=30000,
                max_cpu_hours=100.0,
                max_memory_gb_hours=200.0,
                max_storage_gb=10.0,
                max_bandwidth_gb=100.0,
                max_api_calls_per_minute=60,
                max_agents=5,
                priority_level=5
            )
        elif billing_model == BillingModel.ENTERPRISE:
            return ResourceQuota(
                max_concurrent_jobs=100,
                max_daily_requests=10000,
                max_monthly_requests=300000,
                max_cpu_hours=1000.0,
                max_memory_gb_hours=2000.0,
                max_storage_gb=100.0,
                max_bandwidth_gb=1000.0,
                max_api_calls_per_minute=600,
                max_agents=50,
                priority_level=10
            )
        else:  # PAY_PER_USE
            return ResourceQuota(
                max_concurrent_jobs=5,
                max_daily_requests=500,
                max_monthly_requests=15000,
                max_cpu_hours=50.0,
                max_memory_gb_hours=100.0,
                max_storage_gb=5.0,
                max_bandwidth_gb=50.0,
                max_api_calls_per_minute=30,
                max_agents=3,
                priority_level=3
            )

    async def check_quota(self, tenant_id: str, resource_type: str, amount: float = 1.0) -> bool:
        """Check if tenant has quota available for a resource."""
        if tenant_id not in self.tenants:
            return False

        tenant = self.tenants[tenant_id]
        usage = self.tenant_usage.get(tenant_id)

        if not usage:
            return True

        quota = tenant.quota

        # Check specific resource quotas
        if resource_type == "concurrent_jobs":
            current_jobs = self.resource_allocations.get(tenant_id, {}).get("concurrent_jobs", 0)
            return current_jobs + amount <= quota.max_concurrent_jobs

        elif resource_type == "daily_requests":
            return usage.total_requests + amount <= quota.max_daily_requests

        elif resource_type == "monthly_requests":
            return usage.total_requests + amount <= quota.max_monthly_requests

        elif resource_type == "cpu_hours":
            return usage.cpu_hours_used + amount <= quota.max_cpu_hours

        elif resource_type == "memory_gb_hours":
            return usage.memory_gb_hours_used + amount <= quota.max_memory_gb_hours

        elif resource_type == "storage_gb":
            return usage.storage_gb_used + amount <= quota.max_storage_gb

        elif resource_type == "bandwidth_gb":
            return usage.bandwidth_gb_used + amount <= quota.max_bandwidth_gb

        elif resource_type == "api_calls":
            return self._check_rate_limit(tenant_id, amount)

        return True

    def _check_rate_limit(self, tenant_id: str, calls: int = 1) -> bool:
        """Check API rate limit for tenant."""
        if tenant_id not in self.tenants:
            return False

        quota = self.tenants[tenant_id].quota
        current_time = datetime.now()

        # Get current minute window
        minute_key = current_time.strftime("%Y-%m-%d-%H-%M")

        # Initialize counters if needed
        if tenant_id not in self.rate_limit_counters:
            self.rate_limit_counters[tenant_id] = {}
            self.rate_limit_windows[tenant_id] = {}

        # Clean old windows
        for window_key in list(self.rate_limit_counters[tenant_id].keys()):
            window_time = self.rate_limit_windows[tenant_id].get(window_key)
            if window_time and current_time - window_time > timedelta(minutes=2):
                del self.rate_limit_counters[tenant_id][window_key]
                del self.rate_limit_windows[tenant_id][window_key]

        # Check current window
        current_calls = self.rate_limit_counters[tenant_id].get(minute_key, 0)
        if current_calls + calls > quota.max_api_calls_per_minute:
            return False

        # Update counter
        self.rate_limit_counters[tenant_id][minute_key] = current_calls + calls
        self.rate_limit_windows[tenant_id][minute_key] = current_time

        return True

    async def allocate_resource(self, tenant_id: str, resource_type: str, amount: float) -> bool:
        """Allocate resources to a tenant."""
        if not await self.check_quota(tenant_id, resource_type, amount):
            return False

        # Update allocation
        if tenant_id not in self.resource_allocations:
            self.resource_allocations[tenant_id] = {}

        current_allocation = self.resource_allocations[tenant_id].get(resource_type, 0)
        self.resource_allocations[tenant_id][resource_type] = current_allocation + amount

        # Update usage metrics
        await self._update_usage(tenant_id, resource_type, amount)

        return True

    async def release_resource(self, tenant_id: str, resource_type: str, amount: float):
        """Release allocated resources."""
        if tenant_id in self.resource_allocations:
            current_allocation = self.resource_allocations[tenant_id].get(resource_type, 0)
            new_allocation = max(0, current_allocation - amount)
            self.resource_allocations[tenant_id][resource_type] = new_allocation

    async def _update_usage(self, tenant_id: str, resource_type: str, amount: float):
        """Update usage metrics for a tenant."""
        if tenant_id not in self.tenant_usage:
            return

        usage = self.tenant_usage[tenant_id]

        if resource_type == "daily_requests" or resource_type == "monthly_requests":
            usage.total_requests += int(amount)
        elif resource_type == "cpu_hours":
            usage.cpu_hours_used += amount
        elif resource_type == "memory_gb_hours":
            usage.memory_gb_hours_used += amount
        elif resource_type == "storage_gb":
            usage.storage_gb_used += amount
        elif resource_type == "bandwidth_gb":
            usage.bandwidth_gb_used += amount
        elif resource_type == "api_calls":
            usage.api_calls += int(amount)

        # Calculate cost
        cost = self._calculate_cost(resource_type, amount)
        usage.total_cost += cost

    def _calculate_cost(self, resource_type: str, amount: float) -> float:
        """Calculate cost for resource usage."""
        if resource_type == "cpu_hours":
            return amount * self.pricing_config["cpu_hour_cost"]
        elif resource_type == "memory_gb_hours":
            return amount * self.pricing_config["memory_gb_hour_cost"]
        elif resource_type == "storage_gb":
            return amount * self.pricing_config["storage_gb_cost"]
        elif resource_type == "bandwidth_gb":
            return amount * self.pricing_config["bandwidth_gb_cost"]
        elif resource_type == "api_calls":
            return amount * self.pricing_config["api_call_cost"]
        else:
            return amount * self.pricing_config["job_base_cost"]

    async def _usage_tracking_loop(self):
        """Background task for tracking usage metrics."""
        while self.is_running:
            try:
                await self._collect_usage_metrics()
                await asyncio.sleep(300)  # Update every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in usage tracking loop: {e}")
                await asyncio.sleep(60)

    async def _quota_enforcement_loop(self):
        """Background task for enforcing quotas."""
        while self.is_running:
            try:
                await self._enforce_quotas()
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in quota enforcement loop: {e}")
                await asyncio.sleep(60)

    async def _cleanup_loop(self):
        """Background task for cleanup operations."""
        while self.is_running:
            try:
                await self._cleanup_expired_tenants()
                await self._cleanup_old_usage_data()
                await asyncio.sleep(3600)  # Cleanup every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

    async def _collect_usage_metrics(self):
        """Collect and update usage metrics."""
        for tenant_id, tenant in self.tenants.items():
            try:
                # Reset daily counters if needed
                usage = self.tenant_usage.get(tenant_id)
                if usage:
                    current_time = datetime.now()
                    if current_time.date() > usage.period_start.date():
                        # Start new daily period
                        usage.period_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                        usage.total_requests = 0  # Reset daily counter

                # Save usage to history
                if usage:
                    self.usage_history[tenant_id].append(usage)

                    # Keep only last 30 days of history
                    cutoff_date = datetime.now() - timedelta(days=30)
                    self.usage_history[tenant_id] = [
                        u for u in self.usage_history[tenant_id]
                        if u.period_start > cutoff_date
                    ]

            except Exception as e:
                self.logger.error(f"Error collecting metrics for tenant {tenant_id}: {e}")

    async def _enforce_quotas(self):
        """Enforce quota limits and suspend violating tenants."""
        for tenant_id, tenant in self.tenants.items():
            try:
                if tenant.status != TenantStatus.ACTIVE:
                    continue

                usage = self.tenant_usage.get(tenant_id)
                if not usage:
                    continue

                quota = tenant.quota
                violations = []

                # Check quota violations
                if usage.total_requests > quota.max_daily_requests:
                    violations.append("daily_requests")

                if usage.cpu_hours_used > quota.max_cpu_hours:
                    violations.append("cpu_hours")

                if usage.memory_gb_hours_used > quota.max_memory_gb_hours:
                    violations.append("memory_hours")

                if usage.storage_gb_used > quota.max_storage_gb:
                    violations.append("storage")

                if usage.bandwidth_gb_used > quota.max_bandwidth_gb:
                    violations.append("bandwidth")

                # Take action on violations
                if violations:
                    self.logger.warning(f"Quota violations for tenant {tenant_id}: {violations}")

                    # For now, just log. In production, you might:
                    # - Suspend the tenant
                    # - Send notifications
                    # - Apply throttling
                    # - Charge overage fees

            except Exception as e:
                self.logger.error(f"Error enforcing quotas for tenant {tenant_id}: {e}")

    async def _cleanup_expired_tenants(self):
        """Clean up expired tenants."""
        current_time = datetime.now()
        expired_tenants = []

        for tenant_id, tenant in self.tenants.items():
            if tenant.expires_at and current_time > tenant.expires_at:
                if tenant.status != TenantStatus.EXPIRED:
                    tenant.status = TenantStatus.EXPIRED
                    expired_tenants.append(tenant_id)

        for tenant_id in expired_tenants:
            self.logger.info(f"Tenant {tenant_id} has expired")
            # In production, you might want to:
            # - Send expiration notifications
            # - Archive tenant data
            # - Stop all running jobs

    async def _cleanup_old_usage_data(self):
        """Clean up old usage data."""
        cutoff_date = datetime.now() - timedelta(days=90)

        for tenant_id in list(self.usage_history.keys()):
            self.usage_history[tenant_id] = [
                usage for usage in self.usage_history[tenant_id]
                if usage.period_start > cutoff_date
            ]

    # Database operations

    async def _load_tenants(self):
        """Load tenants from database."""
        try:
            # This would load from your actual database
            # For now, we'll use Redis as a simple storage
            tenants_data = await self.redis_service.get_cache("tenants:all")
            if tenants_data:
                for tenant_data in tenants_data:
                    tenant = self._deserialize_tenant(tenant_data)
                    if tenant:
                        self.tenants[tenant.tenant_id] = tenant

                        # Initialize usage metrics
                        self.tenant_usage[tenant.tenant_id] = UsageMetrics(
                            tenant_id=tenant.tenant_id,
                            period_start=datetime.now(),
                            period_end=datetime.now() + timedelta(days=30)
                        )

            self.logger.info(f"Loaded {len(self.tenants)} tenants")

        except Exception as e:
            self.logger.error(f"Failed to load tenants: {e}")

    async def _save_tenants(self):
        """Save all tenants to database."""
        try:
            tenants_data = [self._serialize_tenant(tenant) for tenant in self.tenants.values()]
            await self.redis_service.set_cache("tenants:all", tenants_data, ttl=None)
            self.logger.debug("Saved all tenants to database")

        except Exception as e:
            self.logger.error(f"Failed to save tenants: {e}")

    async def _save_tenant(self, tenant: TenantConfig):
        """Save a single tenant to database."""
        try:
            tenant_data = self._serialize_tenant(tenant)
            await self.redis_service.set_cache(f"tenant:{tenant.tenant_id}", tenant_data, ttl=None)

        except Exception as e:
            self.logger.error(f"Failed to save tenant {tenant.tenant_id}: {e}")

    async def _delete_tenant_from_db(self, tenant_id: str):
        """Delete tenant from database."""
        try:
            # Delete individual tenant
            await self.redis_service.delete_cache(f"tenant:{tenant_id}")

            # Update the all tenants cache
            await self._save_tenants()

        except Exception as e:
            self.logger.error(f"Failed to delete tenant {tenant_id} from database: {e}")

    def _serialize_tenant(self, tenant: TenantConfig) -> Dict[str, Any]:
        """Serialize tenant configuration to dictionary."""
        data = asdict(tenant)
        data['status'] = tenant.status.value
        data['billing_model'] = tenant.billing_model.value
        data['created_at'] = tenant.created_at.isoformat()
        if tenant.expires_at:
            data['expires_at'] = tenant.expires_at.isoformat()
        return data

    def _deserialize_tenant(self, data: Dict[str, Any]) -> Optional[TenantConfig]:
        """Deserialize tenant configuration from dictionary."""
        try:
            # Convert datetime strings back to datetime objects
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            if data.get('expires_at'):
                data['expires_at'] = datetime.fromisoformat(data['expires_at'])

            # Convert enum strings back to enums
            data['status'] = TenantStatus(data['status'])
            data['billing_model'] = BillingModel(data['billing_model'])

            # Reconstruct quota object
            quota_data = data.pop('quota')
            data['quota'] = ResourceQuota(**quota_data)

            return TenantConfig(**data)

        except Exception as e:
            self.logger.error(f"Failed to deserialize tenant data: {e}")
            return None

    # Public API methods

    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration."""
        return self.tenants.get(tenant_id)

    def get_all_tenants(self) -> Dict[str, TenantConfig]:
        """Get all tenant configurations."""
        return self.tenants.copy()

    def get_tenant_usage(self, tenant_id: str) -> Optional[UsageMetrics]:
        """Get current usage metrics for a tenant."""
        return self.tenant_usage.get(tenant_id)

    def get_tenant_usage_history(self, tenant_id: str, days: int = 30) -> List[UsageMetrics]:
        """Get usage history for a tenant."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            usage for usage in self.usage_history.get(tenant_id, [])
            if usage.period_start > cutoff_date
        ]

    def get_tenant_stats(self) -> Dict[str, Any]:
        """Get overall tenant statistics."""
        total_tenants = len(self.tenants)
        active_tenants = sum(1 for t in self.tenants.values() if t.status == TenantStatus.ACTIVE)
        trial_tenants = sum(1 for t in self.tenants.values() if t.status == TenantStatus.TRIAL)
        suspended_tenants = sum(1 for t in self.tenants.values() if t.status == TenantStatus.SUSPENDED)

        billing_models = defaultdict(int)
        for tenant in self.tenants.values():
            billing_models[tenant.billing_model.value] += 1

        total_revenue = sum(usage.total_cost for usage in self.tenant_usage.values())

        return {
            "total_tenants": total_tenants,
            "active_tenants": active_tenants,
            "trial_tenants": trial_tenants,
            "suspended_tenants": suspended_tenants,
            "billing_models": dict(billing_models),
            "total_revenue": total_revenue,
            "average_revenue_per_tenant": total_revenue / max(total_tenants, 1)
        }

    def is_tenant_authorized(self, tenant_id: str, domain: str = None) -> bool:
        """Check if tenant is authorized to access the system."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        # Check tenant status
        if tenant.status not in [TenantStatus.ACTIVE, TenantStatus.TRIAL]:
            return False

        # Check expiration
        if tenant.expires_at and datetime.now() > tenant.expires_at:
            return False

        # Check domain restrictions
        if domain and tenant.allowed_domains:
            if domain not in tenant.allowed_domains:
                return False

        return True

    def get_tenant_priority(self, tenant_id: str) -> int:
        """Get tenant priority level."""
        tenant = self.tenants.get(tenant_id)
        return tenant.quota.priority_level if tenant else 1
