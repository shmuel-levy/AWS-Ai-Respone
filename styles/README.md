# Styles Directory

This directory contains all CSS styling files for the Document Q&A System.

## File Structure

### `main.css`
Main stylesheet containing:
- Base styles and typography
- Component styling (buttons, inputs, cards)
- Layout and responsive design
- Utility classes
- Status messages and loading states
- Dark mode support

### `rtl.css`
Right-to-Left language support stylesheet containing:
- Hebrew and Arabic text support
- RTL layout adjustments
- RTL form elements
- RTL navigation and tables
- RTL utility classes
- Responsive RTL design

## Usage

The CSS files are automatically loaded by the Streamlit application in `app.py` through the `load_css_files()` function.

## Features

### Main CSS Features
- Modern gradient designs
- Responsive breakpoints
- Utility classes for spacing and alignment
- Professional color scheme
- Smooth transitions and animations
- Accessibility considerations

### RTL CSS Features
- Complete Hebrew text support
- Mixed language handling
- RTL form elements
- RTL navigation menus
- RTL tables and lists
- Mobile-responsive RTL design

## Customization

To modify styles:
1. Edit the appropriate CSS file
2. The changes will be automatically applied when the Streamlit app reloads
3. For production, consider minifying the CSS files

## Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile browsers
- RTL language support in all major browsers
