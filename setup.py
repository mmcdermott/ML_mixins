import setuptools

with open("README.md", "r") as fh: long_description = fh.read()

setuptools.setup(
    name="ml_mixins_mmd", # Replace with your own username
    version="0.0.1",
    author="Matthew McDermott",
    author_email="mattmcdermott8@gmail.com",
    description="Various ML / data-science Mixins",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmcdermott/ML_mixins",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
      'numpy',
    ],
    python_requires='>=3.7',
    test_suite='tests',
)
