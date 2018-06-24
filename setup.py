from setuptools import setup, find_packages

setup(
    name='tf-contextual_prediction_with_expert_advice',
    version='0.0.1',
    license='',
    packages=find_packages(),
    install_requires=[
        'setuptools >= 20.2.2',
        # tensorflow or tensorflow-gpu >= v1.8
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
