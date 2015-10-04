import os

from setuptools import setup

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as buf:
        return buf.read()


conf = dict(
        name='colorcut',
        version='0.1',
        description='Detect colored points and cut images',
        long_description=read('README.md'),
        author='insomnialab',
        author_email='insomnialab@hacari.org',
        url='https://github.com/insomnia-lab/SplitByColor',
        license='AGPL',
        packages=['colorcut'],
        install_requires=[
            'numpy'
        ],
        zip_safe=True,
        entry_points={'console_scripts': [
            'colordetect=colorcut.cdect:main',
            'imagecut=colorcut.cut:main'
        ]},
        classifiers=[
          "License :: OSI Approved :: GNU Affero General Public License v3",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python :: 2",
          "Development Status :: 4 - Beta"
        ])

if __name__ == '__main__':
    setup(**conf)
