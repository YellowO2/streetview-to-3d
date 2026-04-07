# for testings

from streetlevel import streetview

pano = streetview.find_panorama(46.883958, 12.169002)
streetview.download_panorama(pano, f"{pano.id}.jpg")