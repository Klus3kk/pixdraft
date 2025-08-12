"""
Performance Data Collection and Storage System for PixTrick
Stores all benchmark results for trend analysis and ML training
"""

import json
import sqlite3
import hashlib
import platform
import psutil
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import uuid

@dataclass
class SystemInfo:
    """System configuration for reproducible benchmarks"""
    os: str
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    python_version: str
    rust_version: str
    gpu_info: Optional[str]
    
    @classmethod
    def collect(cls):
        """Collect current system information"""
        import subprocess
        
        try:
            # Get Rust version
            rust_version = subprocess.check_output(['rustc', '--version'], 
                                                 text=True).strip()
        except:
            rust_version = "unknown"
        
        try:
            # Basic GPU detection (could be expanded)
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_info = gpus[0].name if gpus else "No GPU"
        except:
            gpu_info = "Unknown GPU"
        
        return cls(
            os=f"{platform.system()} {platform.release()}",
            cpu_model=platform.processor() or "Unknown CPU",
            cpu_cores=psutil.cpu_count(logical=False),
            cpu_threads=psutil.cpu_count(logical=True),
            ram_gb=round(psutil.virtual_memory().total / (1024**3), 1),
            python_version=platform.python_version(),
            rust_version=rust_version,
            gpu_info=gpu_info
        )

@dataclass
class ImageMetadata:
    """Metadata about the test image"""
    width: int
    height: int
    channels: int
    file_size_mb: Optional[float]
    format: Optional[str]
    complexity_score: Optional[float]  # Could measure edge density, etc.
    
    def pixel_count(self) -> int:
        return self.width * self.height
    
    def megapixels(self) -> float:
        return self.pixel_count() / 1_000_000

@dataclass
class FilterConfig:
    """Configuration of applied filters"""
    filter_type: str
    parameters: Dict[str, float]
    enabled: bool
    order_index: int

@dataclass
class PerformanceMetrics:
    """Detailed performance measurements"""
    execution_time_ms: float
    memory_peak_mb: float
    memory_allocated_mb: float
    cpu_usage_percent: float
    cache_hits: int
    cache_misses: int
    
    # Quality metrics (if available)
    output_quality_score: Optional[float] = None
    
    # Detailed timing breakdown
    filter_times_ms: Optional[Dict[str, float]] = None

@dataclass
class BenchmarkResult:
    """Complete benchmark result"""
    # Unique identifiers
    test_id: str
    timestamp: datetime
    commit_hash: Optional[str]
    
    # System context
    system_info: SystemInfo
    
    # Test configuration
    image_metadata: ImageMetadata
    filter_chain: List[FilterConfig]
    
    # Results
    performance_metrics: PerformanceMetrics
    
    # Additional context
    test_type: str  # "manual", "ci", "automated", "user_session"
    tags: List[str]  # ["optimization", "regression_test", "user_workflow"]
    notes: Optional[str]

class PerformanceDatabase:
    """SQLite database for storing performance results"""
    
    def __init__(self, db_path: str = "benchmarks/performance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                test_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                commit_hash TEXT,
                test_type TEXT,
                tags TEXT,  -- JSON array
                notes TEXT,
                
                -- System info
                os TEXT,
                cpu_model TEXT,
                cpu_cores INTEGER,
                cpu_threads INTEGER,
                ram_gb REAL,
                python_version TEXT,
                rust_version TEXT,
                gpu_info TEXT,
                
                -- Image metadata
                img_width INTEGER,
                img_height INTEGER,
                img_channels INTEGER,
                img_megapixels REAL,
                img_file_size_mb REAL,
                img_complexity_score REAL,
                
                -- Performance metrics
                execution_time_ms REAL,
                memory_peak_mb REAL,
                memory_allocated_mb REAL,
                cpu_usage_percent REAL,
                cache_hits INTEGER,
                cache_misses INTEGER,
                output_quality_score REAL,
                
                -- Filter configuration (JSON)
                filter_chain TEXT,
                filter_times_ms TEXT  -- JSON object
            )
        ''')
        
        # Indexes for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmark_results(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_type ON benchmark_results(test_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_megapixels ON benchmark_results(img_megapixels)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_execution_time ON benchmark_results(execution_time_ms)')
        
        conn.commit()
        conn.close()
    
    def store_result(self, result: BenchmarkResult):
        """Store a benchmark result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the actual number of columns in the table
        cursor.execute("PRAGMA table_info(benchmark_results)")
        columns = cursor.fetchall()
        expected_columns = len(columns)
        
        # Prepare values - exactly match the table structure
        values = [
            result.test_id,
            result.timestamp.isoformat(),
            result.commit_hash,
            result.test_type,
            json.dumps(result.tags),
            result.notes,
            
            result.system_info.os,
            result.system_info.cpu_model,
            result.system_info.cpu_cores,
            result.system_info.cpu_threads,
            result.system_info.ram_gb,
            result.system_info.python_version,
            result.system_info.rust_version,
            result.system_info.gpu_info,
            
            result.image_metadata.width,
            result.image_metadata.height,
            result.image_metadata.channels,
            result.image_metadata.megapixels(),
            result.image_metadata.file_size_mb,
            result.image_metadata.complexity_score,
            
            result.performance_metrics.execution_time_ms,
            result.performance_metrics.memory_peak_mb,
            result.performance_metrics.memory_allocated_mb,
            result.performance_metrics.cpu_usage_percent,
            result.performance_metrics.cache_hits,
            result.performance_metrics.cache_misses,
            result.performance_metrics.output_quality_score,
            
            json.dumps([asdict(f) for f in result.filter_chain]),
            json.dumps(result.performance_metrics.filter_times_ms) if result.performance_metrics.filter_times_ms else None
        ]
        
        actual_values = len(values)
        
        # Debug information
        if actual_values != expected_columns:
            print(f"   Column mismatch: table has {expected_columns} columns, providing {actual_values} values")
            print("Table columns:")
            for i, col in enumerate(columns):
                print(f"  {i+1:2d}. {col[1]}")
            print(f"Values being inserted: {actual_values}")
            
            # Try to fix by padding with None if we have too few values
            while len(values) < expected_columns:
                values.append(None)
                print(f"  Added None value #{len(values)}")
        
        # Create the correct number of placeholders
        placeholders = ', '.join(['?'] * len(values))
        
        try:
            cursor.execute(f'''
                INSERT OR REPLACE INTO benchmark_results VALUES ({placeholders})
            ''', values)
            
            conn.commit()
            
        except Exception as e:
            print(f"   Database insert failed: {e}")
            print(f"   Expected columns: {expected_columns}")
            print(f"   Actual values: {len(values)}")
            raise
        finally:
            conn.close()
    
    def query_results(self, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     test_type: Optional[str] = None,
                     min_megapixels: Optional[float] = None,
                     max_megapixels: Optional[float] = None) -> List[Dict]:
        """Query benchmark results with filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        query = "SELECT * FROM benchmark_results WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if test_type:
            query += " AND test_type = ?"
            params.append(test_type)
        
        if min_megapixels:
            query += " AND img_megapixels >= ?"
            params.append(min_megapixels)
        
        if max_megapixels:
            query += " AND img_megapixels <= ?"
            params.append(max_megapixels)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, List]:
        """Get performance trends over time"""
        from datetime import timedelta
        
        start_date = datetime.now() - timedelta(days=days)
        results = self.query_results(start_date=start_date)
        
        # Group by date and calculate statistics
        daily_stats = {}
        for result in results:
            date = result['timestamp'][:10]  # YYYY-MM-DD
            if date not in daily_stats:
                daily_stats[date] = []
            daily_stats[date].append(result['execution_time_ms'])
        
        # Calculate trends
        trends = {
            'dates': list(daily_stats.keys()),
            'avg_times': [sum(times)/len(times) for times in daily_stats.values()],
            'min_times': [min(times) for times in daily_stats.values()],
            'max_times': [max(times) for times in daily_stats.values()],
        }
        
        return trends

class BenchmarkCollector:
    """Collect and store benchmark results"""
    
    def __init__(self, db: PerformanceDatabase):
        self.db = db
        self.system_info = SystemInfo.collect()
    
    def create_benchmark_result(self, 
                              image_metadata: ImageMetadata,
                              filter_chain: List[FilterConfig],
                              performance_metrics: PerformanceMetrics,
                              test_type: str = "manual",
                              tags: List[str] = None,
                              notes: str = None) -> BenchmarkResult:
        """Create a benchmark result object"""
        
        # Generate unique test ID
        test_id = str(uuid.uuid4())
        
        # Try to get git commit hash
        commit_hash = None
        try:
            import subprocess
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                text=True).strip()[:8]
        except:
            pass
        
        return BenchmarkResult(
            test_id=test_id,
            timestamp=datetime.now(),
            commit_hash=commit_hash,
            system_info=self.system_info,
            image_metadata=image_metadata,
            filter_chain=filter_chain,
            performance_metrics=performance_metrics,
            test_type=test_type,
            tags=tags or [],
            notes=notes
        )
    
    def store_benchmark(self, result: BenchmarkResult):
        """Store benchmark result in database"""
        self.db.store_result(result)
        print(f"Stored benchmark result: {result.test_id}")

# Example usage and integration
def example_usage():
    """Example of how to use the performance tracking system"""
    
    # Initialize database
    db = PerformanceDatabase()
    collector = BenchmarkCollector(db)
    
    # Example: Create test data
    image_meta = ImageMetadata(
        width=1920, height=1080, channels=4,
        file_size_mb=5.2, format="PNG",
        complexity_score=0.7
    )
    
    filters = [
        FilterConfig("brightness", {"amount": 20.0}, True, 0),
        FilterConfig("contrast", {"amount": 15.0}, True, 1),
    ]
    
    performance = PerformanceMetrics(
        execution_time_ms=45.2,
        memory_peak_mb=128.5,
        memory_allocated_mb=95.3,
        cpu_usage_percent=85.2,
        cache_hits=12,
        cache_misses=3,
        filter_times_ms={"brightness": 20.1, "contrast": 25.1}
    )
    
    # Create and store result
    result = collector.create_benchmark_result(
        image_meta, filters, performance,
        test_type="automated",
        tags=["optimization", "filter_chain"],
        notes="Testing new parallel processing"
    )
    
    collector.store_benchmark(result)
    
    # Query results
    recent_results = db.query_results(test_type="automated")
    print(f"Found {len(recent_results)} automated test results")
    
    # Get trends
    trends = db.get_performance_trends(days=7)
    print(f"Performance trends: {trends}")

if __name__ == "__main__":
    example_usage()