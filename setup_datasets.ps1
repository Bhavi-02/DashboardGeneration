# Multi-Dataset Setup Script for Windows PowerShell
# This script helps organize your Excel files into multiple datasets

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Multi-Dataset Setup Helper" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory (should be run from Capstone folder)
$dataFolder = "data"

# Check if data folder exists
if (!(Test-Path $dataFolder)) {
    Write-Host "ERROR: data folder not found!" -ForegroundColor Red
    Write-Host "Please run this script from the Capstone project root directory." -ForegroundColor Yellow
    exit
}

Write-Host "Current data folder contents:" -ForegroundColor Green
Get-ChildItem $dataFolder | Format-Table Name, Length, LastWriteTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Dataset Organization Options" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Create sample dataset structure (recommended for first-time setup)"
Write-Host "2. Move existing files to default dataset"
Write-Host "3. Create custom dataset folders"
Write-Host "4. Exit"
Write-Host ""

$choice = Read-Host "Enter your choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Creating sample dataset structure..." -ForegroundColor Yellow
        
        # Create sample folders
        $dataset1 = Join-Path $dataFolder "default"
        $dataset2 = Join-Path $dataFolder "ecommerce_2024"
        
        New-Item -ItemType Directory -Path $dataset1 -Force | Out-Null
        New-Item -ItemType Directory -Path $dataset2 -Force | Out-Null
        
        Write-Host "✓ Created folder: $dataset1" -ForegroundColor Green
        Write-Host "✓ Created folder: $dataset2" -ForegroundColor Green
        
        # Move existing Excel files to default dataset
        $excelFiles = Get-ChildItem $dataFolder -Filter "*.xlsx" | Where-Object { !$_.PSIsContainer }
        
        if ($excelFiles) {
            Write-Host ""
            Write-Host "Moving existing Excel files to 'default' dataset..." -ForegroundColor Yellow
            
            foreach ($file in $excelFiles) {
                $destination = Join-Path $dataset1 $file.Name
                Move-Item $file.FullName $destination -Force
                Write-Host "  ✓ Moved: $($file.Name)" -ForegroundColor Green
            }
        }
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "✓ Sample structure created!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Your data folder now looks like this:"
        Write-Host ""
        Write-Host "data/"
        Write-Host "├── default/"
        Write-Host "│   ├── Dataset.xlsx"
        Write-Host "│   └── Ecommerce Data 2017-2025.xlsx"
        Write-Host "└── ecommerce_2024/"
        Write-Host "    └── (empty - add your 2024 data here)"
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Add more Excel files to the 'ecommerce_2024' folder"
        Write-Host "2. Restart your server (python main.py)"
        Write-Host "3. Open the dashboard and select your dataset from the dropdown"
        Write-Host ""
    }
    
    "2" {
        Write-Host ""
        $datasetName = Read-Host "Enter dataset name (e.g., 'my_data')"
        
        if ($datasetName) {
            $datasetPath = Join-Path $dataFolder $datasetName
            
            New-Item -ItemType Directory -Path $datasetPath -Force | Out-Null
            Write-Host "✓ Created folder: $datasetPath" -ForegroundColor Green
            
            # Move Excel files
            $excelFiles = Get-ChildItem $dataFolder -Filter "*.xlsx" | Where-Object { !$_.PSIsContainer }
            
            if ($excelFiles) {
                Write-Host ""
                Write-Host "Moving Excel files to '$datasetName' dataset..." -ForegroundColor Yellow
                
                foreach ($file in $excelFiles) {
                    $destination = Join-Path $datasetPath $file.Name
                    Move-Item $file.FullName $destination -Force
                    Write-Host "  ✓ Moved: $($file.Name)" -ForegroundColor Green
                }
                
                Write-Host ""
                Write-Host "✓ All files moved successfully!" -ForegroundColor Green
            } else {
                Write-Host "No Excel files found to move." -ForegroundColor Yellow
            }
        }
    }
    
    "3" {
        Write-Host ""
        Write-Host "Enter dataset folder names (comma-separated):" -ForegroundColor Yellow
        Write-Host "Example: sales_2023,sales_2024,hr_data"
        Write-Host ""
        
        $datasets = Read-Host "Dataset names"
        
        if ($datasets) {
            $datasetList = $datasets -split ","
            
            Write-Host ""
            Write-Host "Creating dataset folders..." -ForegroundColor Yellow
            
            foreach ($ds in $datasetList) {
                $ds = $ds.Trim()
                if ($ds) {
                    $datasetPath = Join-Path $dataFolder $ds
                    New-Item -ItemType Directory -Path $datasetPath -Force | Out-Null
                    Write-Host "  ✓ Created: $ds" -ForegroundColor Green
                }
            }
            
            Write-Host ""
            Write-Host "✓ Dataset folders created!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Next steps:" -ForegroundColor Yellow
            Write-Host "1. Copy your Excel files into the respective folders"
            Write-Host "2. Restart your server (python main.py)"
            Write-Host "3. Datasets will be automatically loaded"
        }
    }
    
    "4" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit
    }
    
    default {
        Write-Host "Invalid choice. Exiting..." -ForegroundColor Red
        exit
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Current Data Folder Structure" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Display the tree structure
Get-ChildItem $dataFolder -Recurse | ForEach-Object {
    $indent = "  " * ($_.FullName.Split('\').Count - $dataFolder.Split('\').Count - 1)
    if ($_.PSIsContainer) {
        Write-Host "$indent├── $($_.Name)/" -ForegroundColor Cyan
    } else {
        Write-Host "$indent    ├── $($_.Name)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup complete! Restart your server to load datasets." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
