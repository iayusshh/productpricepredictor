#!/bin/bash

# ML Product Pricing Challenge 2025 - Complete Pipeline Execution Script
# This script runs the complete end-to-end pipeline from data preprocessing to final submission

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="logs/pipeline_execution_$(date +%Y%m%d_%H%M%S).log"
CONFIG_FILE="${SCRIPT_DIR}/config.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Python version
    if ! python3 --version &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Python version: $PYTHON_VERSION"
    
    # Check required directories exist
    for dir in "dataset" "src" "logs" "models" "embeddings" "cache" "images"; do
        if [ ! -d "$dir" ]; then
            log "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Check required files exist
    if [ ! -f "dataset/train.csv" ]; then
        log_error "Training data file not found: dataset/train.csv"
        exit 1
    fi
    
    if [ ! -f "dataset/test.csv" ]; then
        log_error "Test data file not found: dataset/test.csv"
        exit 1
    fi
    
    # Check if virtual environment exists and activate it
    if [ -d "venv" ]; then
        log "Activating virtual environment..."
        source venv/bin/activate
    else
        log_warning "Virtual environment not found. Using system Python."
    fi
    
    # Install/check dependencies
    if [ -f "requirements.txt" ]; then
        log "Installing/checking Python dependencies..."
        pip install -r requirements.txt --quiet
    fi
    
    log_success "Prerequisites check completed"
}

# Function to run data validation
validate_data() {
    log "Validating input data..."
    
    python3 -c "
import pandas as pd
import sys

try:
    # Validate training data
    train_df = pd.read_csv('dataset/train.csv')
    print(f'Training data: {len(train_df)} samples')
    required_train_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
    missing_train_cols = [col for col in required_train_cols if col not in train_df.columns]
    if missing_train_cols:
        print(f'ERROR: Missing columns in train.csv: {missing_train_cols}')
        sys.exit(1)
    
    # Validate test data
    test_df = pd.read_csv('dataset/test.csv')
    print(f'Test data: {len(test_df)} samples')
    required_test_cols = ['sample_id', 'catalog_content', 'image_link']
    missing_test_cols = [col for col in required_test_cols if col not in test_df.columns]
    if missing_test_cols:
        print(f'ERROR: Missing columns in test.csv: {missing_test_cols}')
        sys.exit(1)
    
    print('Data validation successful')
except Exception as e:
    print(f'ERROR: Data validation failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Data validation completed"
    else
        log_error "Data validation failed"
        exit 1
    fi
}

# Function to run the complete pipeline
run_pipeline() {
    local mode=${1:-"full"}
    local config_arg=""
    
    if [ -f "$CONFIG_FILE" ]; then
        config_arg="--config $CONFIG_FILE"
    fi
    
    log "Starting ML Product Pricing pipeline in $mode mode..."
    
    case $mode in
        "full")
            log "Running complete pipeline (training + prediction)..."
            python3 src/main.py --mode full $config_arg
            ;;
        "train")
            log "Running training pipeline only..."
            python3 src/main.py --mode train $config_arg
            ;;
        "predict")
            log "Running prediction pipeline only..."
            # Find the most recent model
            MODEL_FILE=$(find models -name "*.pkl" -type f -exec ls -t {} + | head -n1)
            if [ -z "$MODEL_FILE" ]; then
                log_error "No trained models found for prediction"
                exit 1
            fi
            log "Using model: $MODEL_FILE"
            python3 src/main.py --mode predict --model-path "$MODEL_FILE" $config_arg
            ;;
        "orchestrator")
            log "Running pipeline orchestrator with error recovery..."
            python3 src/pipeline_orchestrator.py $config_arg
            ;;
        *)
            log_error "Invalid mode: $mode. Use 'full', 'train', 'predict', or 'orchestrator'"
            exit 1
            ;;
    esac
}

# Function to validate outputs
validate_outputs() {
    log "Validating pipeline outputs..."
    
    # Check if output file exists
    if [ ! -f "test_out.csv" ]; then
        log_error "Output file test_out.csv not found"
        return 1
    fi
    
    # Validate output format
    python3 -c "
import pandas as pd
import sys

try:
    # Load output file
    output_df = pd.read_csv('test_out.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    # Check columns
    if list(output_df.columns) != ['sample_id', 'price']:
        print(f'ERROR: Invalid output columns: {list(output_df.columns)}')
        sys.exit(1)
    
    # Check row count
    if len(output_df) != len(test_df):
        print(f'ERROR: Row count mismatch. Output: {len(output_df)}, Test: {len(test_df)}')
        sys.exit(1)
    
    # Check sample_id matching
    if set(output_df['sample_id']) != set(test_df['sample_id']):
        print('ERROR: Sample IDs do not match between output and test data')
        sys.exit(1)
    
    # Check price values
    if output_df['price'].isna().any():
        print('ERROR: Output contains missing price values')
        sys.exit(1)
    
    if (output_df['price'] <= 0).any():
        print('ERROR: Output contains non-positive price values')
        sys.exit(1)
    
    print(f'Output validation successful: {len(output_df)} predictions')
    print(f'Price range: {output_df[\"price\"].min():.2f} - {output_df[\"price\"].max():.2f}')
    print(f'Mean price: {output_df[\"price\"].mean():.2f}')
    
except Exception as e:
    print(f'ERROR: Output validation failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Output validation completed"
        return 0
    else
        log_error "Output validation failed"
        return 1
    fi
}

# Function to generate summary report
generate_summary() {
    log "Generating pipeline summary report..."
    
    SUMMARY_FILE="logs/pipeline_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "ML Product Pricing Challenge 2025 - Pipeline Execution Summary"
        echo "=============================================================="
        echo "Execution Date: $(date)"
        echo "Script Directory: $SCRIPT_DIR"
        echo ""
        
        echo "Data Summary:"
        if [ -f "dataset/train.csv" ]; then
            TRAIN_COUNT=$(wc -l < dataset/train.csv)
            echo "  Training samples: $((TRAIN_COUNT - 1))"
        fi
        
        if [ -f "dataset/test.csv" ]; then
            TEST_COUNT=$(wc -l < dataset/test.csv)
            echo "  Test samples: $((TEST_COUNT - 1))"
        fi
        echo ""
        
        echo "Generated Artifacts:"
        echo "  Models:"
        find models -name "*.pkl" -type f | while read -r file; do
            SIZE=$(du -h "$file" | cut -f1)
            echo "    $file ($SIZE)"
        done
        
        echo "  Logs:"
        find logs -name "*.json" -type f | head -5 | while read -r file; do
            SIZE=$(du -h "$file" | cut -f1)
            echo "    $file ($SIZE)"
        done
        
        if [ -f "test_out.csv" ]; then
            OUTPUT_SIZE=$(du -h test_out.csv | cut -f1)
            OUTPUT_LINES=$(wc -l < test_out.csv)
            echo "  Output: test_out.csv ($OUTPUT_SIZE, $((OUTPUT_LINES - 1)) predictions)"
        fi
        
        echo ""
        echo "Storage Usage:"
        echo "  Total project size: $(du -sh . | cut -f1)"
        echo "  Models directory: $(du -sh models 2>/dev/null | cut -f1 || echo '0B')"
        echo "  Cache directory: $(du -sh cache 2>/dev/null | cut -f1 || echo '0B')"
        echo "  Logs directory: $(du -sh logs 2>/dev/null | cut -f1 || echo '0B')"
        echo "  Images directory: $(du -sh images 2>/dev/null | cut -f1 || echo '0B')"
        
    } > "$SUMMARY_FILE"
    
    log_success "Summary report generated: $SUMMARY_FILE"
    
    # Display summary to console
    cat "$SUMMARY_FILE"
}

# Function to cleanup temporary files
cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove temporary cache files older than 1 day
    find cache -name "*.tmp" -type f -mtime +1 -delete 2>/dev/null || true
    
    # Compress old log files
    find logs -name "*.log" -type f -mtime +7 -exec gzip {} \; 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Function to display help
show_help() {
    cat << EOF
ML Product Pricing Challenge 2025 - Pipeline Execution Script

Usage: $0 [OPTIONS] [MODE]

MODES:
    full         Run complete pipeline (training + prediction) [default]
    train        Run training pipeline only
    predict      Run prediction pipeline only (requires trained models)
    orchestrator Run pipeline orchestrator with error recovery

OPTIONS:
    --config FILE    Use custom configuration file
    --no-cleanup     Skip cleanup of temporary files
    --validate-only  Only validate data and outputs, don't run pipeline
    --help          Show this help message

EXAMPLES:
    $0                          # Run complete pipeline
    $0 train                    # Train models only
    $0 predict                  # Generate predictions only
    $0 orchestrator             # Run with error recovery
    $0 --validate-only          # Validate data and outputs
    $0 --config custom.json full # Use custom configuration

ENVIRONMENT VARIABLES:
    PYTHON_PATH     Path to Python executable (default: python3)
    LOG_LEVEL       Logging level (default: INFO)
    MAX_WORKERS     Maximum number of parallel workers

EOF
}

# Main execution function
main() {
    local mode="full"
    local validate_only=false
    local no_cleanup=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --no-cleanup)
                no_cleanup=true
                shift
                ;;
            --validate-only)
                validate_only=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            full|train|predict|orchestrator)
                mode="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Create log file
    mkdir -p logs
    touch "$LOG_FILE"
    
    log "Starting ML Product Pricing Challenge 2025 Pipeline"
    log "Mode: $mode"
    log "Log file: $LOG_FILE"
    
    # Check prerequisites
    check_prerequisites
    
    # Validate data
    validate_data
    
    if [ "$validate_only" = true ]; then
        log "Validation-only mode - skipping pipeline execution"
        validate_outputs
        exit $?
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run the pipeline
    if run_pipeline "$mode"; then
        log_success "Pipeline execution completed successfully"
        
        # Validate outputs
        if validate_outputs; then
            log_success "All validations passed"
        else
            log_warning "Output validation failed, but pipeline completed"
        fi
        
    else
        log_error "Pipeline execution failed"
        exit 1
    fi
    
    # Calculate execution time
    END_TIME=$(date +%s)
    EXECUTION_TIME=$((END_TIME - START_TIME))
    log "Total execution time: $((EXECUTION_TIME / 60)) minutes $((EXECUTION_TIME % 60)) seconds"
    
    # Generate summary
    generate_summary
    
    # Cleanup if requested
    if [ "$no_cleanup" != true ]; then
        cleanup
    fi
    
    log_success "Pipeline execution completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review the generated test_out.csv file"
    echo "2. Check the methodology documentation"
    echo "3. Verify compliance with competition requirements"
    echo "4. Submit your solution"
}

# Handle script interruption
trap 'log_error "Pipeline execution interrupted"; exit 1' INT TERM

# Run main function
main "$@"