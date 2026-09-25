import base64
import html
import json
import re
import unittest

from streetview_to_3d.ui.viewers import build_pointcloud_viewer, iframe
from streetview_to_3d.visualise.build_viewer import SOURCE, OUTPUT, build_document


class BuildTests(unittest.TestCase):
    def test_packaged_modules_match_sources(self):
        doc = build_document()
        imports = json.loads(re.search(r'<script type="importmap">(.*?)</script>', doc, re.S)[1])["imports"]
        for path in SOURCE.glob("*.js"):
            encoded = imports[f"@viewer/{path.stem}"].split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded), path.read_bytes())
            for dependency in re.findall(r"from ['\"](@viewer/[^'\"]+)", path.read_text()):
                self.assertIn(dependency, imports)
        self.assertEqual(OUTPUT.read_text(), doc)

    def test_gradio_config_and_sandbox(self):
        url = "/gradio_api/file=/test/scene.json?unsafe=</script>"
        embed = build_pointcloud_viewer(scene_url=url)
        doc = html.unescape(re.search(r'srcdoc="(.*?)" sandbox=', embed, re.S)[1])
        config = re.search(r'<script id="viewer-config" type="application/json">(.*?)</script>', doc, re.S)[1]
        self.assertEqual(json.loads(config)["sceneUrl"], url)
        self.assertFalse(json.loads(config)["editable"])
        self.assertNotIn("</script>", config)
        self.assertIn("allow-downloads", embed)
        self.assertNotIn("allow-pointer-lock", iframe("map"))
