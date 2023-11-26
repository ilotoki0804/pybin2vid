# pybin2vid

<!-- ![[CI](https://github.com/ilotoki0804/pybin2vid/workflows/ci/badge.svg?branch=main)](https://github.com/ilotoki0804/pybin2vid/actions?workflow=ci)
[![Codecov](https://codecov.io/gh/ilotoki0804/pybin2vid/branch/main/graph/badge.svg)](https://codecov.io/gh/ilotoki0804/pybin2vid)
[![Maintainability](https://api.codeclimate.com/v1/badges/d96cc9a1841a819cd4f5/maintainability)](https://codeclimate.com/github/ilotoki0804/pybin2vid/maintainability)
[![Code Climate technical debt](https://img.shields.io/codeclimate/tech-debt/ilotoki0804/pybin2vid)](https://codeclimate.com/github/ilotoki0804/pybin2vid)
[![Read the Docs](https://img.shields.io/readthedocs/pybin2vid/latest?label=Read%20the%20Docs)](https://pybin2vid.readthedocs.io/en/latest/index.html) -->

Convert Arbitrary Bineries to Video.

이 프로젝트는 임의의 바이너리 데이터를 영상으로 변환할 수 있도록 합니다.

## Motivation

유튜브는 영상을 무료로 매우 많은 양을 올릴 수 있도록 합니다.
만약 임의의 데이터를 영상으로 변경하는 방법이 있다면 유튜브를 데이터를 저장하고 보낼 수 있다는 수단으로 사용할 수 있다는 생각에 계속해서 관련 방식을 찾아보던 중 다른 독립적인 개발자도 개발자도 같은 이유로 관련 프로그램을 만들었다는 사실을 유튜브를 통해 알게 되었습니다.

해당 유튜브 영상의 댓글 중 하나가 자체적인 오류 수정 방식을 가지고 있는 QR 코드를 이용할 수도 있다고 가능성을 제시했고, 다른 댓글은 그 R, G, B에 각각 다른 QR 코드를 인코딩하면 더 많은 양을 인코딩할 수 있다고 제시했습니다.

현재 이 프로젝트는 QR 코드(정확히는 datamatrix)를 이용해 임의의 바이너리 데이터를 QR 코드로 변환하는 것까지 구현되었고, RGB에 각각 다른 데이터를 인코딩하는 방식은 제작 중에 있습니다.

## Dependency

이 프로젝트를 사용하려면 다음과 같은 pypi 패키지 의존성이 필요합니다.

```
pylibdmtx
numpy
pillow
opencv-python
matplotlib
```

다운로드 받으려면 `requirements.txt`를 이용할 수 있습니다.

```console
pip -r requirements.txt
```

또한 pypi 외 의존성으로는 [libdmtx](http://libdmtx.wikidot.com/general-instructions)와 [ffmpeg](https://ffmpeg.org/)가 요구되며 우분투에서는 다음과 같이 다운로드받을 수 있습니다.

```console
sudo apt install libdmtx0b ffmpeg
```

윈도우나 맥OS에서도 이론상 가능하지만, 테스트되지는 않았습니다.

## Example

어떤 바이너리 정보를 영상으로 변환해 `output.mp4`에 저장하고 싶다면 다음과 같이 코드를 짜세요.

```python
from pathlib import Path
from bin2vid import encode_to_video, decode_from_video

data = Path("test_data/constitution_law.txt").read_bytes()

encode_to_video(data, "output.mp4")
```

만약 이미지 폴더를 끝난 후 삭제하고 싶다면 다음과 같이 코드를 짜세요.

```python
encode_to_video(data, "output.mp4", delete_images_folder_after_finished=True)
```

영상으로 저장된 데이터를 불러오고 싶다면 다음과 같이 코드를 짜세요.

```python
from bin2vid import encode_to_video, decode_from_video

decode_from_video("output.mp4")
```

## Limitation

이 방식으로 만들어진 데이터는 어째선지 유튜브에 업로드되지 않습니다.

바이너리 데이터를 영상으로 변환할 때 파일 크기는 약 10배로 늘어나고, 약 150배 가량의 여유 저장공간이 요구됩니다. 예를 들어 4MB짜리 바이너리를 영상으로 만든다면 40MB 가량의 영상이 나오고, 이 과정에서 약 600MB 가량의 저장 공간이 필요합니다. 필요한 저장공간이 영상 크기보다 큰 이유는 만드는 과정에서 프레임으로 사용할 이미지들을 각각 제작하기 때문입니다.

## Changelog

v0.1.0: 시작
