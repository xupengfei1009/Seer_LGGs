# Seer_LGGs

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

## Overview

Seer_LGGs is a Python-based application developed with Streamlit, focusing on prediction analysis for Low-Grade Gliomas (LGGs). This project integrates machine learning models to provide a diagnostic assistance tool for medical professionals.

## Features

- Interactive web interface built with Streamlit
- Integrated pre-trained model (DCPHModelFinal.pkl)
- User-friendly interface
- Quick prediction results output

## Requirements

The project uses `environment.yml` for dependency management. Main requirements include:

```yaml
dependencies:
  - python>=3.7
  - streamlit
  - pandas
  - scikit-learn
  - pickle
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/xupengfei1009/Seer_LGGs.git
cd Seer_LGGs
```

2. Create and activate the environment
```bash
conda env create -f environment.yml
conda activate seer-lggs
```

3. Run the application
```bash
streamlit run streamlit_app.py
```

## Usage

1. After launching the application, access it through your browser (default address: http://localhost:8501)
2. Input the required parameters following the interface prompts
3. The system will generate predictions based on the input data

## Development Environment

The project includes `.devcontainer` configuration, supporting VS Code's remote development container feature to ensure consistency in development environments.

## Contributing

Contributions through Issues and Pull Requests are welcome. Before submitting, please ensure:

1. Code follows Python coding standards
2. Necessary comments and documentation are added
3. All tests pass

## License

[MIT License](LICENSE)

## Contact

- Author: Pengfei Xu
- GitHub: [@xupengfei1009](https://github.com/xupengfei1009)

## Acknowledgments

Thanks to all developers and researchers who have contributed to this project.
