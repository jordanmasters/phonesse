from setuptools import setup


  # def readme():
  #     with open('README.rst') as f:
  #         return f.read()


setup(name='phonesse',
      version='0.1',
      description='Sound Search Regex Interface',
      # long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='',
      url='',
      author='Verbal',
      author_email='',
      license='MIT',
      packages=['phonesse'],
      install_requires=[
          'markdown',
          'pronouncing'
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      scripts=['bin/phonesse'],
      include_package_data=True,
      zip_safe=False)