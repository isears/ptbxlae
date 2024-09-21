import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ptbxlae",
    version="0.0.1",
    author="Isaac Sears",
    author_email="isaac.j.sears@gmail.com",
    description="PTB-XL Autoencoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["ptbxlae"],
    url="https://github.com/isears/ptbxlae",
    project_urls={
        "Bug Tracker": "https://github.com/isears/ptbxlae/issues",
    },
)
