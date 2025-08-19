# 🎨 Styles Directory - Document Q&A System

## 📁 File Structure

```
styles/
├── main.css          # Main stylesheet with mobile-responsive design
└── README.md         # This documentation file
```

## 🎯 Main Stylesheet (`main.css`)

### **Features:**
- ✅ **Mobile-First Design** - Responsive layout for all devices
- ✅ **Modern UI Components** - Professional button and input styling
- ✅ **Accessibility Support** - Focus indicators and high contrast mode
- ✅ **Cross-Platform Compatibility** - Works on desktop, tablet, and mobile
- ✅ **Print Styles** - Optimized for printing

### **Design Philosophy:**
- **Clean & Professional** - Modern gradient buttons and subtle shadows
- **Mobile-Optimized** - Touch-friendly buttons and responsive layouts
- **Accessibility First** - Proper focus states and contrast ratios
- **Performance Focused** - Efficient CSS with minimal overhead

### **Key Components:**

#### **Header & Typography**
- Gradient text effects for main headers
- Responsive font sizing for different screen sizes
- Professional color scheme (blues and oranges)

#### **Interactive Elements**
- Hover effects on buttons with smooth transitions
- Focus states for keyboard navigation
- Touch-friendly sizing for mobile devices

#### **Layout & Responsiveness**
- Mobile-first approach with progressive enhancement
- Flexible column layouts that adapt to screen size
- Optimized sidebar behavior on mobile devices

#### **Status Messages**
- Color-coded alerts for different message types
- Consistent styling across success, warning, and error states
- Professional appearance with subtle shadows

### **Mobile Responsiveness:**

#### **Breakpoints:**
- **Mobile**: `max-width: 768px` - Single column layout, touch-optimized
- **Tablet**: `769px - 1024px` - Balanced layout with medium sizing
- **Desktop**: `min-width: 1025px` - Full layout with enhanced hover effects

#### **Mobile Features:**
- Full-width buttons and inputs
- Collapsible sidebar with info button
- Optimized touch targets (minimum 44px)
- Responsive typography scaling

### **Browser Support:**
- **Modern Browsers**: Chrome, Firefox, Safari, Edge (latest versions)
- **Mobile Browsers**: iOS Safari, Chrome Mobile, Samsung Internet
- **Fallbacks**: Graceful degradation for older browsers

## 🚀 Usage

### **In Your Streamlit App:**
```python
def load_css_files():
    """Load external CSS files for styling."""
    try:
        with open('styles/main.css', 'r', encoding='utf-8') as f:
            main_css = f.read()
        
        st.markdown(f"""
            <style>
            {main_css}
            </style>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading CSS files: {e}")
```

### **Customization:**
- **Colors**: Modify CSS variables in the `:root` section
- **Layouts**: Adjust breakpoints in media queries
- **Components**: Customize specific element styles

## 🔧 Development

### **Adding New Styles:**
1. Follow the existing CSS structure and naming conventions
2. Use BEM methodology for class naming
3. Include responsive variants for mobile and desktop
4. Test across different screen sizes

### **CSS Organization:**
- **Base Styles** - Global resets and typography
- **Component Styles** - Buttons, inputs, containers
- **Layout Styles** - Grid systems and positioning
- **Responsive Styles** - Media queries and breakpoints
- **Utility Classes** - Helper classes for common patterns

## 📱 Mobile Optimization

### **Touch-Friendly Design:**
- Minimum 44px touch targets
- Adequate spacing between interactive elements
- Smooth scrolling and transitions
- Optimized for thumb navigation

### **Performance:**
- Minimal CSS overhead
- Efficient selectors and properties
- Optimized animations and transitions
- Fast loading on mobile networks

---

## 🎉 Benefits

This stylesheet provides:
- **Professional Appearance** - Clean, modern design that impresses users
- **Mobile Excellence** - Perfect experience on all devices
- **Accessibility** - Inclusive design for all users
- **Maintainability** - Well-organized, documented CSS
- **Performance** - Fast loading and smooth interactions

The mobile-first approach ensures your app looks great and works perfectly on any device! 📱✨
