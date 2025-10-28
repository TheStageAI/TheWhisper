from pathlib import Path
from setuptools import find_packages, setup


ROOT_DIR = Path(__file__).parent


def read_readme() -> str:
    readme_path = ROOT_DIR / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


def read_requirements() -> list[str]:
    requirements_path = ROOT_DIR / "requirements.txt"
    if not requirements_path.exists():
        return []
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    requirements: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirements.append(stripped)
    return requirements


setup(
    name="thestage-asr",
    version="0.1.0",
    description="Optimized Whisper models for streaming and on-device use",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("examples*", "scripts*", "tests*")),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: FastAPI",
    ],
    keywords=["asr", "whisper", "streaming", "mlx", "nvidia", "fastapi"],
    license="MIT",
    url="https://github.com/a31m/TheWhisper",
    project_urls={
        "Source": "https://github.com/a31m/TheWhisper",
        "Issues": "https://github.com/a31m/TheWhisper/issues",
    },
)

# TODO: add elastic_models as dependency
# TODO: add conditional install pip install thestage-asr[nvidia] or pip install thestage-asr[apple]
