#!/usr/bin/env python3
"""
Phase 3: Enterprise & Scalability Features - Comprehensive Demo

This script demonstrates all the enterprise-grade features implemented in Phase 3:
- Distributed Scraping Architecture
- Advanced Queue Management
- Multi-tenant Support
- Horizontal Scaling
- Enhanced Health Monitoring

Run this script to see the complete enterprise system in action.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phase3_demo")

# Import Phase 3 components
from services.enterprise_orchestrator import EnterpriseOrchestrator, EnterpriseConfig
from services.multi_tenant_manager import BillingModel, ResourceQuota
from services.horizontal_scaling_manager import LoadBalancingAlgorithm
from services.enhanced_health_monitor import FailoverStrategy
from services.advanced_queue_manager import QueueType, ResourceRequirement


class Phase3EnterpriseDemo:
    """Comprehensive demonstration of Phase 3 enterprise features."""
    
    def __init__(self):
        self.orchestrator: EnterpriseOrchestrator = None
        self.demo_tenants: Dict[str, str] = {}
        self.demo_jobs: Dict[str, str] = {}
        
    async def run_complete_demo(self):
        """Run the complete Phase 3 enterprise demonstration."""
        logger.info("🚀 Starting Phase 3: Enterprise & Scalability Features Demo")
        logger.info("=" * 80)
        
        try:
            # 1. Initialize Enterprise System
            await self.demo_system_initialization()
            
            # 2. Demonstrate Cluster Management
            await self.demo_cluster_management()
            
            # 3. Demonstrate Multi-Tenant Features
            await self.demo_multi_tenant_features()
            
            # 4. Demonstrate Advanced Queue Management
            await self.demo_advanced_queue_management()
            
            # 5. Demonstrate Horizontal Scaling
            await self.demo_horizontal_scaling()
            
            # 6. Demonstrate Health Monitoring
            await self.demo_health_monitoring()
            
            # 7. Demonstrate Enterprise Job Processing
            await self.demo_enterprise_job_processing()
            
            # 8. Show System Metrics and Status
            await self.demo_system_metrics()
            
            logger.info("✅ Phase 3 Enterprise Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup()
    
    async def demo_system_initialization(self):
        """Demonstrate enterprise system initialization."""
        logger.info("\n🏢 1. Enterprise System Initialization")
        logger.info("-" * 50)
        
        # Create enterprise configuration
        config = EnterpriseConfig(
            cluster_id="demo-enterprise-cluster",
            cluster_name="Demo Enterprise Web Scraper",
            max_nodes=20,
            auto_scaling=True,
            multi_tenant_enabled=True,
            scaling_enabled=True,
            health_monitoring_enabled=True,
            default_billing_model=BillingModel.FREEMIUM,
            load_balancing_algorithm=LoadBalancingAlgorithm.LEAST_LOADED,
            failover_strategy=FailoverStrategy.GRACEFUL
        )
        
        logger.info(f"📋 Cluster ID: {config.cluster_id}")
        logger.info(f"📋 Max Nodes: {config.max_nodes}")
        logger.info(f"📋 Multi-tenant: {config.multi_tenant_enabled}")
        logger.info(f"📋 Auto-scaling: {config.auto_scaling}")
        
        # Initialize orchestrator
        self.orchestrator = EnterpriseOrchestrator(config)
        
        # Start the system
        logger.info("🔄 Starting enterprise orchestrator...")
        success = await self.orchestrator.start()
        
        if success:
            logger.info("✅ Enterprise system started successfully")
            
            # Show initial status
            status = self.orchestrator.get_system_status()
            logger.info(f"📊 System Status: {status['system']['running']}")
            logger.info(f"📊 Cluster Health: {status['cluster']['cluster_health']:.2f}")
        else:
            raise Exception("Failed to start enterprise system")
    
    async def demo_cluster_management(self):
        """Demonstrate cluster management features."""
        logger.info("\n🌐 2. Distributed Cluster Management")
        logger.info("-" * 50)
        
        # Get cluster information
        cluster_stats = self.orchestrator.cluster_manager.get_cluster_stats()
        logger.info(f"📊 Cluster ID: {cluster_stats['cluster_id']}")
        logger.info(f"📊 Total Nodes: {cluster_stats['total_nodes']}")
        logger.info(f"📊 Healthy Nodes: {cluster_stats['healthy_nodes']}")
        logger.info(f"📊 Leader Node: {cluster_stats['leader_node_id']}")
        logger.info(f"📊 Discovery Method: {cluster_stats['discovery_method']}")
        
        # Show node details
        nodes = self.orchestrator.cluster_manager.get_cluster_nodes()
        for node_id, node_info in nodes.items():
            logger.info(f"🖥️  Node {node_id[:8]}... - {node_info.role.value} - {node_info.status.value}")
            if node_info.resources:
                cpu = node_info.resources.get('cpu_usage', 0)
                memory = node_info.resources.get('memory_usage', 0)
                logger.info(f"   💻 CPU: {cpu:.1f}%, Memory: {memory:.1f}%")
    
    async def demo_multi_tenant_features(self):
        """Demonstrate multi-tenant features."""
        logger.info("\n🏢 3. Multi-Tenant Support")
        logger.info("-" * 50)
        
        # Create different types of tenants
        tenants_to_create = [
            {
                "name": "Startup Corp",
                "email": "admin@startup.com",
                "billing_model": BillingModel.FREEMIUM,
                "quota": None
            },
            {
                "name": "Growth Company",
                "email": "admin@growth.com",
                "billing_model": BillingModel.SUBSCRIPTION,
                "quota": ResourceQuota(
                    max_concurrent_jobs=25,
                    max_daily_requests=5000,
                    max_cpu_hours=200.0,
                    priority_level=5
                )
            },
            {
                "name": "Enterprise Client",
                "email": "admin@enterprise.com",
                "billing_model": BillingModel.ENTERPRISE,
                "quota": ResourceQuota(
                    max_concurrent_jobs=100,
                    max_daily_requests=50000,
                    max_cpu_hours=1000.0,
                    priority_level=10
                )
            }
        ]
        
        for tenant_data in tenants_to_create:
            tenant_id = await self.orchestrator.create_tenant(
                name=tenant_data["name"],
                email=tenant_data["email"],
                billing_model=tenant_data["billing_model"],
                custom_quota=tenant_data["quota"]
            )
            
            if tenant_id:
                self.demo_tenants[tenant_data["name"]] = tenant_id
                logger.info(f"✅ Created tenant: {tenant_data['name']} ({tenant_data['billing_model'].value})")
                logger.info(f"   🆔 Tenant ID: {tenant_id}")
                
                # Show tenant quota
                tenant_config = self.orchestrator.multi_tenant_manager.get_tenant(tenant_id)
                if tenant_config:
                    quota = tenant_config.quota
                    logger.info(f"   📊 Max Jobs: {quota.max_concurrent_jobs}")
                    logger.info(f"   📊 Daily Requests: {quota.max_daily_requests}")
                    logger.info(f"   📊 Priority Level: {quota.priority_level}")
        
        # Show tenant statistics
        tenant_stats = self.orchestrator.multi_tenant_manager.get_tenant_stats()
        logger.info(f"\n📈 Tenant Statistics:")
        logger.info(f"   Total Tenants: {tenant_stats['total_tenants']}")
        logger.info(f"   Active Tenants: {tenant_stats['active_tenants']}")
        logger.info(f"   Total Revenue: ${tenant_stats['total_revenue']:.2f}")
    
    async def demo_advanced_queue_management(self):
        """Demonstrate advanced queue management."""
        logger.info("\n📊 4. Advanced Queue Management")
        logger.info("-" * 50)
        
        # Create specialized queue partitions
        partitions_to_create = [
            {
                "partition_id": "high_priority",
                "name": "High Priority Queue",
                "queue_type": QueueType.PRIORITY,
                "max_size": 5000,
                "priority_weight": 3.0
            },
            {
                "partition_id": "batch_processing",
                "name": "Batch Processing Queue",
                "queue_type": QueueType.FIFO,
                "max_size": 10000,
                "priority_weight": 0.5
            },
            {
                "partition_id": "enterprise_queue",
                "name": "Enterprise Queue",
                "queue_type": QueueType.WEIGHTED,
                "max_size": 15000,
                "priority_weight": 5.0
            }
        ]
        
        for partition_data in partitions_to_create:
            success = self.orchestrator.queue_manager.create_partition(**partition_data)
            if success:
                logger.info(f"✅ Created partition: {partition_data['name']}")
                logger.info(f"   🆔 ID: {partition_data['partition_id']}")
                logger.info(f"   📊 Type: {partition_data['queue_type'].value}")
                logger.info(f"   📊 Max Size: {partition_data['max_size']}")
        
        # Show queue statistics
        queue_stats = self.orchestrator.queue_manager.get_queue_statistics()
        logger.info(f"\n📈 Queue Statistics:")
        logger.info(f"   Total Partitions: {queue_stats['global']['total_partitions']}")
        logger.info(f"   Pending Tasks: {queue_stats['global']['total_pending']}")
        logger.info(f"   Running Tasks: {queue_stats['global']['total_running']}")
        logger.info(f"   Completed Tasks: {queue_stats['global']['total_completed']}")
        
        # Show resource status
        resource_status = self.orchestrator.queue_manager.get_resource_status()
        logger.info(f"\n💻 Resource Status:")
        for resource_type, status in resource_status['resources'].items():
            logger.info(f"   {resource_type}: {status['utilization_percent']:.1f}% used")
    
    async def demo_horizontal_scaling(self):
        """Demonstrate horizontal scaling features."""
        logger.info("\n⚡ 5. Horizontal Scaling & Load Balancing")
        logger.info("-" * 50)
        
        # Show current scaling policies
        scaling_policies = self.orchestrator.scaling_manager.get_scaling_policies()
        logger.info(f"📋 Active Scaling Policies: {len(scaling_policies)}")
        
        for policy_id, policy in scaling_policies.items():
            logger.info(f"   📊 {policy.name}")
            logger.info(f"      Metric: {policy.metric_name}")
            logger.info(f"      Scale Up: >{policy.scale_up_threshold}")
            logger.info(f"      Scale Down: <{policy.scale_down_threshold}")
            logger.info(f"      Node Range: {policy.min_nodes}-{policy.max_nodes}")
        
        # Show load balancer statistics
        lb_stats = self.orchestrator.scaling_manager.get_load_balancer_stats()
        logger.info(f"\n🔄 Load Balancer Statistics:")
        logger.info(f"   Algorithm: {lb_stats['algorithm']}")
        logger.info(f"   Total Nodes: {lb_stats['total_nodes']}")
        logger.info(f"   Total Connections: {lb_stats['total_connections']}")
        logger.info(f"   Sticky Sessions: {lb_stats['sticky_sessions']}")
        
        # Show scaling recommendations
        recommendations = self.orchestrator.scaling_manager.get_scaling_recommendations()
        logger.info(f"\n📈 Scaling Recommendations:")
        for rec in recommendations:
            logger.info(f"   📊 {rec['policy_name']}: {rec['recommendation']}")
            logger.info(f"      Current: {rec['current_nodes']} nodes")
            logger.info(f"      Suggested: {rec['suggested_nodes']} nodes")
            logger.info(f"      Metric Value: {rec['current_value']:.1f}")
    
    async def demo_health_monitoring(self):
        """Demonstrate health monitoring and failover."""
        logger.info("\n🔍 6. Enhanced Health Monitoring")
        logger.info("-" * 50)
        
        # Show health monitoring statistics
        health_stats = self.orchestrator.health_monitor.get_monitoring_stats()
        logger.info(f"📊 Monitoring Statistics:")
        logger.info(f"   Total Health Checks: {health_stats['total_health_checks']}")
        logger.info(f"   Failed Nodes: {health_stats['failed_nodes_count']}")
        logger.info(f"   Recovery Queue: {health_stats['recovery_queue_count']}")
        logger.info(f"   Average Health Score: {health_stats['average_health_score']:.1f}")
        
        # Show node health scores
        health_scores = self.orchestrator.health_monitor.get_all_health_scores()
        logger.info(f"\n💚 Node Health Scores:")
        for node_id, score in health_scores.items():
            logger.info(f"   🖥️  Node {node_id[:8]}...")
            logger.info(f"      Overall Score: {score.overall_score:.1f}/100")
            logger.info(f"      Availability: {score.availability_score:.1f}%")
            logger.info(f"      Performance: {score.performance_score:.1f}/100")
            logger.info(f"      Resources: {score.resource_score:.1f}/100")
            logger.info(f"      Trend: {score.health_trend}")
        
        # Show circuit breaker states
        circuit_breakers = health_stats.get('circuit_breakers', {})
        if circuit_breakers:
            logger.info(f"\n🔌 Circuit Breaker States:")
            for node_id, state in circuit_breakers.items():
                logger.info(f"   Node {node_id[:8]}...: {state}")
        
        # Show recent failover events
        failover_history = self.orchestrator.health_monitor.get_failover_history(hours=1)
        logger.info(f"\n🔄 Recent Failover Events: {len(failover_history)}")
        for event in failover_history[-3:]:  # Show last 3 events
            logger.info(f"   📅 {event.timestamp.strftime('%H:%M:%S')}")
            logger.info(f"      Node: {event.failed_node_id[:8]}...")
            logger.info(f"      Reason: {event.trigger_reason}")
            logger.info(f"      Success: {event.success}")
    
    async def demo_enterprise_job_processing(self):
        """Demonstrate enterprise job processing with different tenants."""
        logger.info("\n🚀 7. Enterprise Job Processing")
        logger.info("-" * 50)
        
        # Submit jobs for different tenants with different priorities
        job_scenarios = [
            {
                "tenant": "Startup Corp",
                "task_data": {"url": "https://example.com/startup", "extract": ["title"]},
                "priority": 3,
                "partition": "default",
                "description": "Basic scraping job"
            },
            {
                "tenant": "Growth Company",
                "task_data": {"url": "https://example.com/growth", "extract": ["title", "content"]},
                "priority": 6,
                "partition": "high_priority",
                "description": "Priority scraping job"
            },
            {
                "tenant": "Enterprise Client",
                "task_data": {"url": "https://example.com/enterprise", "extract": ["title", "content", "metadata"]},
                "priority": 9,
                "partition": "enterprise_queue",
                "description": "Enterprise scraping job",
                "resource_requirements": ResourceRequirement(
                    cpu_cores=2.0,
                    memory_mb=1024,
                    estimated_duration=600
                )
            }
        ]
        
        for scenario in job_scenarios:
            tenant_name = scenario["tenant"]
            tenant_id = self.demo_tenants.get(tenant_name)
            
            if tenant_id:
                job_id = await self.orchestrator.submit_job(
                    task_data=scenario["task_data"],
                    tenant_id=tenant_id,
                    priority=scenario["priority"],
                    resource_requirements=scenario.get("resource_requirements"),
                    partition_id=scenario["partition"]
                )
                
                if job_id:
                    self.demo_jobs[scenario["description"]] = job_id
                    logger.info(f"✅ Submitted: {scenario['description']}")
                    logger.info(f"   🆔 Job ID: {job_id}")
                    logger.info(f"   👤 Tenant: {tenant_name}")
                    logger.info(f"   📊 Priority: {scenario['priority']}")
                    logger.info(f"   📂 Partition: {scenario['partition']}")
                else:
                    logger.warning(f"❌ Failed to submit job for {tenant_name}")
        
        # Wait a moment for job processing
        await asyncio.sleep(2)
        
        # Show updated queue statistics
        queue_stats = self.orchestrator.queue_manager.get_queue_statistics()
        logger.info(f"\n📈 Updated Queue Statistics:")
        logger.info(f"   Pending Tasks: {queue_stats['global']['total_pending']}")
        logger.info(f"   Running Tasks: {queue_stats['global']['total_running']}")
        
        # Show tenant usage
        for tenant_name, tenant_id in self.demo_tenants.items():
            usage = self.orchestrator.multi_tenant_manager.get_tenant_usage(tenant_id)
            if usage:
                logger.info(f"\n💰 {tenant_name} Usage:")
                logger.info(f"   Total Requests: {usage.total_requests}")
                logger.info(f"   Total Cost: ${usage.total_cost:.4f}")
    
    async def demo_system_metrics(self):
        """Show comprehensive system metrics."""
        logger.info("\n📊 8. System Metrics & Performance")
        logger.info("-" * 50)
        
        # Get comprehensive system status
        status = self.orchestrator.get_system_status()
        
        logger.info(f"🏢 System Overview:")
        logger.info(f"   Cluster: {status['system']['cluster_name']}")
        logger.info(f"   Uptime: {status['system']['uptime_seconds']:.0f} seconds")
        logger.info(f"   Running: {status['system']['running']}")
        
        logger.info(f"\n🌐 Cluster Status:")
        logger.info(f"   Health: {status['cluster']['cluster_health']:.2f}")
        logger.info(f"   Total Nodes: {status['cluster']['total_nodes']}")
        logger.info(f"   Healthy Nodes: {status['cluster']['healthy_nodes']}")
        
        logger.info(f"\n📊 Queue Performance:")
        logger.info(f"   Total Pending: {status['queues']['global']['total_pending']}")
        logger.info(f"   Total Running: {status['queues']['global']['total_running']}")
        logger.info(f"   Total Completed: {status['queues']['global']['total_completed']}")
        
        if 'tenants' in status:
            logger.info(f"\n🏢 Tenant Overview:")
            logger.info(f"   Total Tenants: {status['tenants']['total_tenants']}")
            logger.info(f"   Active Tenants: {status['tenants']['active_tenants']}")
            logger.info(f"   Total Revenue: ${status['tenants']['total_revenue']:.2f}")
        
        # Get detailed performance metrics
        metrics = self.orchestrator.get_performance_metrics()
        logger.info(f"\n📈 Performance Metrics:")
        logger.info(f"   Total Requests: {metrics['system_metrics']['total_requests']}")
        logger.info(f"   Successful Requests: {metrics['system_metrics']['successful_requests']}")
        logger.info(f"   Failed Requests: {metrics['system_metrics']['failed_requests']}")
        logger.info(f"   Active Nodes: {metrics['system_metrics']['active_nodes']}")
    
    async def cleanup(self):
        """Clean up demo resources."""
        logger.info("\n🧹 Cleaning up demo resources...")
        
        if self.orchestrator:
            try:
                await self.orchestrator.stop()
                logger.info("✅ Enterprise orchestrator stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping orchestrator: {e}")


async def main():
    """Main demo function."""
    demo = Phase3EnterpriseDemo()
    
    try:
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("\n👋 Phase 3 Enterprise Demo finished")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
