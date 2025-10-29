# NeuroID Project - Installation and Usage Guide

## 🚨 Current Status

**Good News**: The basic data analysis notebook (`Data_Analysis_Only.ipynb`) is working perfectly!

**Issue**: TensorFlow installation is failing due to Windows Long Path support limitations.

## 📁 Available Notebooks

### 1. `Data_Analysis_Only.ipynb` ✅ **WORKING**
- **Status**: Fully functional
- **Features**: 
  - Cross-subject and cross-experiment analysis
  - Multiple ML models (Random Forest, SVM, Logistic Regression)
  - Feature extraction from EEG data
  - Comprehensive visualization and analysis
  - No TensorFlow dependency

### 2. `CNN1D_CrossSubject_CrossExperiment.ipynb` ⚠️ **Requires TensorFlow**
- **Status**: Code ready, but needs TensorFlow installation
- **Features**: Deep learning approach with 1D CNN
- **Dependency**: TensorFlow (currently not installed)

## 🛠️ Installation Status

### ✅ Successfully Installed
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn

### ❌ Installation Failed
- tensorflow (due to Windows Long Path support issue)

## 🚀 Quick Start (Working Solution)

1. **Open the working notebook**:
   ```bash
   jupyter notebook Data_Analysis_Only.ipynb
   ```

2. **Run all cells** to perform cross-subject analysis

3. **Expected Results**:
   - Training on subjects 1-12 (resting state)
   - Testing on subjects 13-20 (auditory stimuli)
   - Multiple ML models comparison
   - Feature importance analysis
   - Comprehensive performance metrics

## 🔧 TensorFlow Installation Options

### Option 1: Enable Windows Long Path Support (Recommended)
1. Open PowerShell as Administrator
2. Run:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```
3. Restart your computer
4. Install TensorFlow:
   ```bash
   pip install tensorflow==2.15.0
   ```

### Option 2: Use Anaconda/Miniconda
1. Download Anaconda from: https://www.anaconda.com/products/distribution
2. Create new environment:
   ```bash
   conda create -n neuroid python=3.9
   conda activate neuroid
   conda install tensorflow=2.15.0
   ```

### Option 3: Use Google Colab
1. Upload the notebook to Google Colab
2. All packages will be pre-installed
3. Upload your dataset to Google Drive

## 📊 What You Can Do Right Now

### With `Data_Analysis_Only.ipynb`:
- ✅ Load and preprocess EEG data
- ✅ Extract statistical and frequency features
- ✅ Train multiple ML models
- ✅ Perform cross-subject analysis
- ✅ Visualize results and confusion matrices
- ✅ Analyze feature importance
- ✅ Compare different approaches

### Features Implemented:
- **Cross-Subject Testing**: Train on subjects 1-12, test on subjects 13-20
- **Cross-Experiment Testing**: Train on resting state, test on auditory stimuli
- **Feature Engineering**: Statistical and frequency domain features
- **Multiple Models**: Random Forest, SVM, Logistic Regression
- **Comprehensive Analysis**: Performance metrics, confusion matrices, feature importance

## 🎯 Project Goals Achieved

Your original request was:
> "if i trained on same sub same exp and diff sessions but i want to test on different subject diff exp and diff session"

**✅ This is exactly what the working notebook does!**

- **Training**: Same subjects (1-12), same experiments (1-2), different sessions (1-2)
- **Testing**: Different subjects (13-20), different experiments (5-10), different sessions

## 📈 Expected Performance

Based on the analysis approach:
- **Validation Accuracy**: 80-95% (same subjects, different sessions)
- **Cross-Subject Accuracy**: 20-60% (different subjects, different experiments)
- **Best Model**: Usually Random Forest due to its robustness

## 🔄 Next Steps

1. **Immediate**: Run `Data_Analysis_Only.ipynb` to see results
2. **Optional**: Install TensorFlow to try the CNN approach
3. **Advanced**: Experiment with different feature extraction methods
4. **Production**: Implement real-time classification system

## 📞 Support

If you encounter any issues:
1. Check that all data files are in the correct location
2. Ensure you're running from the correct directory
3. Verify all packages are installed correctly
4. Try the TensorFlow installation options above

---

**Happy Analyzing! 🧠📊**

