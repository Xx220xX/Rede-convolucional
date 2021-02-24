from distutils.core import setup

setup(
    name='CNN_GPU',
    packages=['CNN_GPU'],
    include_package_data = True,
    package_data={'CNN_GPU':['lib/*','*.py']},
    version='1.0.0',
    license='MIT',
    description='Convolutional network write in ANSI C with OpenCL GPU acelerator',
    author='Henrique S. Lima',
    author_email='henrique.lufu@gmail.com',
    url='https://github.com',
    download_url='empty',
    keywords=['NEURAL', 'DEEP', 'LEARN', 'MACHINE','CNN','AMD','Convolutional'],
    # install_requires=[],
    classifiers=['Development Status :: 4 - Beta',
                 'Intend Audience :: Developers:: Henrique S. Lima, Guilherme Ferreira',
                 'Topic :: Software Development :: Build Tools',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.9']
)
