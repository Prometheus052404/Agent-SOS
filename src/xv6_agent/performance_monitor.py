"""
Performance Monitor Module - Performance tracking for Xv6 agent.

Implements:
- Context manager for operation timing
- P95 latency tracking
- Threshold breach alerts
- Metrics export to JSON
"""

import time
import json
import logging
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class OperationMetric:
    """Metric for a single operation."""
    name: str
    duration_ms: float
    timestamp: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Threshold configuration for an operation."""
    warning_ms: float
    critical_ms: float


class PerformanceMonitor:
    """
    Monitors performance of agent operations.
    """

    def __init__(
        self,
        metrics_file: str = ".xv6_agent/metrics.json",
        thresholds: Optional[Dict[str, PerformanceThreshold]] = None
    ):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Default thresholds
        self.thresholds = thresholds or {
            'cpg_update': PerformanceThreshold(warning_ms=500, critical_ms=1000),
            'context_assembly': PerformanceThreshold(warning_ms=200, critical_ms=500),
            'llm_response': PerformanceThreshold(warning_ms=30000, critical_ms=60000),
            'vector_search': PerformanceThreshold(warning_ms=100, critical_ms=500),
            'file_sentinel': PerformanceThreshold(warning_ms=200, critical_ms=500)
        }
        
        # In-memory metrics
        self.metrics: Dict[str, List[OperationMetric]] = defaultdict(list)
        self.max_history = 1000  # Keep last 1000 metrics per operation
        
        # Load existing metrics
        self._load_metrics()

    def _load_metrics(self):
        """Load existing metrics from disk."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                
                for name, ops in data.get('metrics', {}).items():
                    for op in ops[-self.max_history:]:
                        self.metrics[name].append(OperationMetric(
                            name=op['name'],
                            duration_ms=op['duration_ms'],
                            timestamp=op['timestamp'],
                            success=op.get('success', True),
                            metadata=op.get('metadata', {})
                        ))
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")

    def _save_metrics(self):
        """Save metrics to disk."""
        data = {
            'last_updated': datetime.now().isoformat(),
            'metrics': {
                name: [
                    {
                        'name': m.name,
                        'duration_ms': m.duration_ms,
                        'timestamp': m.timestamp,
                        'success': m.success,
                        'metadata': m.metadata
                    }
                    for m in ops[-100:]  # Save last 100 per operation
                ]
                for name, ops in self.metrics.items()
            }
        }
        
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    @contextmanager
    def measure(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager to measure operation duration.
        
        Usage:
            with monitor.measure('cpg_update'):
                # do work
        """
        start_time = time.time()
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            metric = OperationMetric(
                name=operation_name,
                duration_ms=duration_ms,
                timestamp=datetime.now().isoformat(),
                success=success,
                metadata=metadata or {}
            )
            
            self._record_metric(metric)
            self._check_threshold(metric)

    def _record_metric(self, metric: OperationMetric):
        """Record a metric."""
        self.metrics[metric.name].append(metric)
        
        # Trim history
        if len(self.metrics[metric.name]) > self.max_history:
            self.metrics[metric.name] = self.metrics[metric.name][-self.max_history:]
        
        # Periodically save (every 10 metrics)
        total_metrics = sum(len(ops) for ops in self.metrics.values())
        if total_metrics % 10 == 0:
            self._save_metrics()

    def _check_threshold(self, metric: OperationMetric):
        """Check if metric exceeds thresholds."""
        threshold = self.thresholds.get(metric.name)
        
        if not threshold:
            return
        
        if metric.duration_ms >= threshold.critical_ms:
            logger.warning(
                f"CRITICAL: {metric.name} took {metric.duration_ms:.0f}ms "
                f"(threshold: {threshold.critical_ms}ms)"
            )
        elif metric.duration_ms >= threshold.warning_ms:
            logger.info(
                f"WARNING: {metric.name} took {metric.duration_ms:.0f}ms "
                f"(threshold: {threshold.warning_ms}ms)"
            )

    def get_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for an operation."""
        ops = self.metrics.get(operation_name, [])
        
        if not ops:
            return {'count': 0}
        
        durations = [op.duration_ms for op in ops]
        
        return {
            'count': len(durations),
            'min_ms': min(durations),
            'max_ms': max(durations),
            'mean_ms': statistics.mean(durations),
            'median_ms': statistics.median(durations),
            'p95_ms': self._percentile(durations, 95),
            'p99_ms': self._percentile(durations, 99),
            'success_rate': sum(1 for op in ops if op.success) / len(ops)
        }

    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1
        
        if c >= len(sorted_data):
            return sorted_data[-1]
        
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations."""
        return {
            name: self.get_stats(name)
            for name in self.metrics.keys()
        }

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent threshold breaches."""
        alerts = []
        
        for name, ops in self.metrics.items():
            threshold = self.thresholds.get(name)
            if not threshold:
                continue
            
            for op in ops[-50:]:  # Check last 50
                if op.duration_ms >= threshold.warning_ms:
                    alerts.append({
                        'operation': op.name,
                        'duration_ms': op.duration_ms,
                        'timestamp': op.timestamp,
                        'level': 'critical' if op.duration_ms >= threshold.critical_ms else 'warning'
                    })
        
        # Sort by timestamp and return most recent
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        return alerts[:limit]

    def export_report(self, output_file: Optional[str] = None) -> str:
        """Export a performance report."""
        output_file = output_file or str(self.metrics_file.parent / "performance_report.json")
        
        report = {
            'generated': datetime.now().isoformat(),
            'statistics': self.get_all_stats(),
            'recent_alerts': self.get_recent_alerts(),
            'thresholds': {
                name: {'warning_ms': t.warning_ms, 'critical_ms': t.critical_ms}
                for name, t in self.thresholds.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file


# Global monitor instance
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def measure(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Convenience decorator/context manager for measuring operations."""
    return get_monitor().measure(operation_name, metadata)


if __name__ == "__main__":
    # Test the performance monitor
    logging.basicConfig(level=logging.DEBUG)
    
    monitor = PerformanceMonitor()
    
    # Simulate some operations
    for i in range(5):
        with monitor.measure('cpg_update'):
            time.sleep(0.1 + i * 0.05)  # Simulate varying latency
    
    for i in range(3):
        with monitor.measure('vector_search'):
            time.sleep(0.05)
    
    # Simulate a slow operation
    with monitor.measure('context_assembly'):
        time.sleep(0.3)
    
    # Print stats
    print("\n=== Performance Stats ===")
    stats = monitor.get_all_stats()
    for name, s in stats.items():
        print(f"\n{name}:")
        print(f"  Count: {s['count']}")
        print(f"  Mean: {s['mean_ms']:.1f}ms")
        print(f"  P95: {s['p95_ms']:.1f}ms")
    
    # Print alerts
    print("\n=== Recent Alerts ===")
    alerts = monitor.get_recent_alerts()
    for alert in alerts:
        print(f"  [{alert['level']}] {alert['operation']}: {alert['duration_ms']:.0f}ms")
