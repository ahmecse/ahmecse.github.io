export const bio = {
  about: {
    text: [
      "Hi &#128075;",
      "I'm Ahmed Shmels Muhe. I’m currently working as Machine Learning Inten at Suvidha Foundation (Suvidha Mahila Mandal) specialize NLP,ML and deep Learning Applications. I did my undergrad in CS from VIT Vellore.",
      "I'm a developer, geek and curious human besides being an Data Science enthusiast. I have experience of delivering solutions for interesting problems from startup and industry space.",
      "I love to read, Code, and travel.",
    ],
  },
  contact: {
    text: [
      "If you would like to get in touch with me, be it for exploring a technology, a business, or to just say hi, feel free to send me an email. My email address is: ahmecse@gmail.com",
    ],
  },
};

export const skills = [
  {
    title: "Languages",
    skillName: "Python",
    color: "1",
    percentage: "80",
  },
  {
    title: "Languages",
    skillName: "JavaScript, Java",
    color: "1",
    percentage: "70",
  },
  {
    title: "Frameworks/Libraries",
    skillName: "Angular, Reactjs",
    color: "6",
    percentage: "50",
  },
  {
    title: "Backend",
    skillName: "Nodejs, MongoDB",
    color: "2",
    percentage: "40",
  },
  {
    title: "Clouds",
    skillName: "AWS(EC2, S3), Heroku, Netlify",
    color: "3",
    percentage: "30",
  },
  {
    title: "Design",
    skillName: "HTML,CSS, Bootstrap, SCSS",
    color: "4",
    percentage: "80",
  },
  {
    title: "Version Control",
    skillName: "GitHub, JIRA, Trello",
    color: "7",
    percentage: "70",
  },
  {
    title: "Tools",
    skillName: "Postman, Chrome DevTools",
    color: "3",
    percentage: "80",
  },
  {
    title: "Saas products",
    skillName: "CleverTap, FreshDesk",
    color: "5",
    percentage: "50",
  },
  {
    title: "Editor",
    skillName: "VS Code",
    color: "6",
    percentage: "70",
  },
];

export const projects = {
  webProjects: [
    {
      projectName: "Programming Diaries",
      image: "images/programmingdiaries.png",
      summary:
        "Developed a full stack blog application to provide content on techical topics across the internet with admin interface.",
      preview: "https://programmingdiaries.herokuapp.com/",
      techStack: ["Django", "SQLite", "Bootstrap", "JavaScript", "Heroku"],
    },
    {
      projectName: "Find Your Bank",
      image: "images/findyourbank.png",
      summary:
        "Developed a React application to render a list of banks fetched from API. Filtered the banks based on queries from localstorage, marked favorites banks.",
      preview: "https://clever-fermi-0d5d76.netlify.app",
      techStack: ["Reactjs", "Bootstrap", "JavaScript", "Netlify"],
    },
    {
      projectName: "Web Portfolio",
      image: "images/portfolio.png",
      summary:
        "Web Portfolio to showcase acadmics, skills, projects and contact details in better manner.",
      preview: "https://github.com/ahmecse",
      techStack: ["HTML", "Bootstrap", "JavaScript"],
    },
    {
      projectName: "Resume Builder",
      image: "images/resume-builder.png",
      summary:
        "Browser based editor to build and download Resumes in a customizable templates.",
      preview: "https://vinaysomawat.github.io/Resume-Builder",
      techStack: ["HTML", "Bootstrap", "JavaScript"],
    },
  ],
  softwareProjects: [
    {
      projectName: "Pizza Ordering ChatBot",
      image: "images/pizzaorderchatbot.png",
      summary:
        "ChatBot using Dialogflow, Firebase database which stores the chat data in the realtime database.",
      preview: "https://github.com/vinaysomawat/Pizza-Ordering-ChatBot",
      techStack: ["Dailogflow", "Firebase"],
    },
    {
      projectName: "WhatsApp-Bot",
      image: "images/whatsappbot.jpg",
      summary:
        "Python script which helps to send messages to WhatsApp contacts automatically using selenium and web automation.",
      preview: "https://github.com/vinaysomawat/WhatsApp-Bot",
      techStack: ["Selenium", "Chrome Webdriver", "Python"],
    },
    {
      projectName: "Bill Generator",
      image: "images/billgenerator.png",
      summary:
        "GUI to transfer data to excel sheets and generate bills on the local shops.",
      preview: "https://github.com/vinaysomawat/Bill-Generator",
      techStack: ["Tkinter", "Openxlpy", "Python"],
    },
  ],
  androidProjects: [
    {
      projectName: "NITW-CSE",
      image: "images/nitwcse.jpg",
      summary:
        "The Application display details of Department courses, reference books, research, publication and faculty profile.",
      preview: "https://github.com/vinaysomawat/NITW-CSE",
      techStack: ["JAVA", "XML", "Android"],
    },
    {
      projectName: "CareerHigh-App",
      image: "images/carrerhigh.png",
      summary:
        "The Application display the webpages of website careerhigh.in in android devices.",
      preview: "https://github.com/vinaysomawat/CareerHigh-Android",
      techStack: ["JAVA", "XML", "Android"],
    },
  ],
  freelanceProjects: [
    {
      projectName: "SnylloAir.com",
      image: "images/snylloair.png",
      summary:
        "Developed a company website to showcase the purpose, services and products provided by the company to audience.",
      preview: "https://www.snylloair.com/",
      techStack: ["Bootstrap", "JavaScript", "AWS-S3"],
    },
    {
      projectName: "Delivery+",
      image: "images/AM-Logo-.png",
      summary: "Android Application to display website in android devices.",
      preview:
        "https://play.google.com/store/apps/details?id=com.americanmarket.americanmarketandroid",
      techStack: ["Android", "JAVA", "Play Store"],
    },
  ],
};

export const experience = [
  {
    title: "Biofourmis India Pvt. Ltd.",
    duration: "April 2022 - Present",
    subtitle: "Software Engineer",
    details: [
      "Working on the products in the healthcare/digital therapeutics domain.",
    ],
    tags: ["JavaScript", "Angular", "Bootstrap", "Nodejs", "Jenkins"],
    icon: "heartbeat",
  },
  {
    title: "Novopay Solutions Pvt. Ltd.",
    duration: "June 2020 - April 2022",
    subtitle: "Software Engineer",
    details: [
      "Implemented Aadhaar Enabled Payment services such as Bio-metric eKYC, Cash Withdrawal, Balance Enquiry, Mini-Statements, and Money transfer; completed more than 20 story points in each sprint.",
      "Integrated QR Code and reduced the effective time by 50 percent to load money into wallet, Clevertap events to track user actions, Freshdesk ticketing service and chat-bot services. Worked on user onboarding, approval, and finance interfaces.",
      "Co-ordinated closely with the product team, backend team, android team, and QA team to deliver the product builds before deadlines.",
    ],
    tags: ["JavaScript", "Angular", "React", "Bootstrap", "Nodejs", "Jenkins"],
    icon: "qrcode",
  },
  {
    title: "ThinkPedia LLP",
    duration: "May 2019 - June 2019",
    subtitle: "SDE Intern",
    details: [
      "Worked as a full stack developer to support tech team.",
      "Developed a customer Web Application from scratch for social media management.",
    ],
    tags: ["JavaScript", "Angular", "Bootstrap", "Java", "Spring Boot"],
    icon: "group",
  },
];

export const education = [
  {
    title: "Bachelors in Computer Science and Engineering",
    duration: "",
    subtitle: "Vellore Institute of Technology,VIT, Vellore",
    details: [
      "Ranked 1st place in Cyber Security Hackton, VIT-2018.",
      "Active Competitive Programmer with CodeChef Rating 1841*.",
      "Received 500+ stars and 300+ forks on GitHub projects.",
    ],
    tags: [
      "Data Structures & Algorithms",
      "Operating Systems",
      "Database Management System",
      "Machine Learning",
      "Artificial Intelligence",
      "Software Engineering",
    ],
    icon: "graduation-cap",
  },
  {
    title: "Class 11-12th in Science Stream",
    duration: "",
    subtitle: "Haik Secondary Education, Haik",
    details: [
      "Qualified Ethiopian General Secondary Education Certificate Examination (EGSECE).",
      "Secured 93.5 percentile in Class 12th.",
    ],
    tags: ["Physics", "Chemistry", "Mathematics"],
    icon: "book",
  },
];

export const footer = [
  {
    label: "Dev Profiles",
    data: [
      {
        text: "Stackoverflow",
        link: "https://meta.stackexchange.com/users/1342771/ahmecse/",
      },
      {
        text: "GitHub",
        link: "https://github.com/ahmecse",
      },
      {
        text: "LeetCode",
        link: "https://leetcode.com/ahmecse/",
      },
    ],
  },
  {
    label: "Resources",
    data: [
      {
        text: "Enable Dark/Light Mode",
        func: "enableDarkMode()",
      },
      {
        text: "Print this page",
        func: "window.print()",
      },
      {
        text: "Clone this page",
        link: "https://ahmecse.github.io/",
      },
    ],
  },
  {
    label: "Social Profiles",
    data: [
      {
        text: "Linkedin",
        link: "https://www.linkedin.com/in/ahmecse/",
      },
      {
        text: "Twitter",
        link: "https://twitter.com/AHMED_SHMELS",
      },
      {
        text: "Buy me a coffee",
        link: "https://www.buymeacoffee.com/ahmecse",
      },
    ],
  },
  {
    label: "copyright-text",
    data: [
      "Made with &hearts; by Ahmed Shmels .",
      "&copy; No Copyrights. Feel free to use this template.",
    ],
  },
];

const gitUserName = 'ahmecse';
const mediumUserName = 'ahmecse';

export const URLs = {
	mediumURL: `https://api.rss2json.com/v1/api.json?rss_url=https://medium.com/feed/@${mediumUserName}`,
};
