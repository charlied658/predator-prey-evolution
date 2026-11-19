"""Check font selection without depending on the host operating system."""
import unittest
from unittest.mock import patch, sentinel
from src.analysis.replay_fonts import replay_font


class ReplayFontChecks(unittest.TestCase):
    def test_windows_keeps_original_font_and_size(self):
        with patch('src.analysis.replay_fonts.ImageFont.truetype', return_value=sentinel.font) as load:
            self.assertIs(replay_font(15), sentinel.font)
            load.assert_called_once_with('C:/Windows/Fonts/segoeui.ttf', 15)

    def test_missing_windows_font_uses_mac_font(self):
        with patch('src.analysis.replay_fonts.ImageFont.truetype', side_effect=[OSError(), sentinel.font]) as load:
            self.assertIs(replay_font(14), sentinel.font)
            self.assertEqual(load.call_args.args, ('/System/Library/Fonts/Supplemental/Arial.ttf', 14))

    def test_missing_system_fonts_uses_pillow_default(self):
        with patch('src.analysis.replay_fonts.ImageFont.truetype', side_effect=OSError()), patch(
            'src.analysis.replay_fonts.ImageFont.load_default', return_value=sentinel.font
        ) as fallback:
            self.assertIs(replay_font(19), sentinel.font)
            fallback.assert_called_once_with(size=19)


if __name__ == '__main__':
    unittest.main()
