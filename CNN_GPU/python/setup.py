from distutils.core import setup

setup(
    name='gabriela_gpu',
    packages=['gabriela_gpu'],
    include_package_data = True,
    package_data={'gabriela_gpu':['lib/*']},
    version='2.4.0005',
    license='MIT',
    description='a ultra fast library for deep learn,can use AMD GPU',
    author='Henrique S. Lima',
    author_email='henrique.lufu@gmail.com',
    url='https://github.com',
    download_url='empty',
    keywords=['NEURAL', 'DEEP', 'LEARN', 'MACHINE','DNN','AMD'],
    # install_requires=[],
    classifiers=['Development Status :: 4 - Beta',
                 'Intend Audience :: Developers',
                 'Topic :: Software Development :: Build Tools',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.7']
)
